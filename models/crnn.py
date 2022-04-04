def create_crnn_model(fingerprint_input, model_settings,
                      model_size_info, is_training):
    """Builds a model with convolutional recurrent networks with GRUs
    Based on the model definition in https://arxiv.org/abs/1703.05390
    model_size_info: defines the following convolution layer parameters
        {number of conv features, conv filter height, width, stride in y,x dir.},
        followed by number of GRU layers and number of GRU cells per layer
    Optionally, the bi-directional GRUs and/or GRU with layer-normalization
      can be explored.
    """
    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])

    layer_norm = False
    bidirectional = False

    # CNN part
    first_filter_count = model_size_info[0]
    first_filter_height = model_size_info[1]
    first_filter_width = model_size_info[2]
    first_filter_stride_y = model_size_info[3]
    first_filter_stride_x = model_size_info[4]

    first_weights = tf.get_variable('W', shape=[first_filter_height,
                                                first_filter_width, 1, first_filter_count],
                                    initializer=tf.contrib.layers.xavier_initializer())

    first_bias = tf.Variable(tf.zeros([first_filter_count]))
    first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [
        1, first_filter_stride_y, first_filter_stride_x, 1
    ], 'VALID') + first_bias
    first_relu = tf.nn.relu(first_conv)
    if is_training:
        first_dropout = tf.nn.dropout(first_relu, dropout_prob)
    else:
        first_dropout = first_relu
    first_conv_output_width = int(math.floor(
        (input_frequency_size - first_filter_width + first_filter_stride_x) /
        first_filter_stride_x))
    first_conv_output_height = int(math.floor(
        (input_time_size - first_filter_height + first_filter_stride_y) /
        first_filter_stride_y))

    # GRU part
    num_rnn_layers = model_size_info[5]
    RNN_units = model_size_info[6]
    flow = tf.reshape(first_dropout, [-1, first_conv_output_height,
                                      first_conv_output_width * first_filter_count])
    cell_fw = []
    cell_bw = []
    if layer_norm:
        for i in range(num_rnn_layers):
            cell_fw.append(LayerNormGRUCell(RNN_units))
            if bidirectional:
                cell_bw.append(LayerNormGRUCell(RNN_units))
    else:
        for i in range(num_rnn_layers):
            cell_fw.append(tf.contrib.rnn.GRUCell(RNN_units))
            if bidirectional:
                cell_bw.append(tf.contrib.rnn.GRUCell(RNN_units))

    if bidirectional:
        outputs, output_state_fw, output_state_bw = \
            tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cell_fw, cell_bw, flow,
                                                           dtype=tf.float32)
        flow_dim = first_conv_output_height * RNN_units * 2
        flow = tf.reshape(outputs, [-1, flow_dim])
    else:
        cells = tf.contrib.rnn.MultiRNNCell(cell_fw)
        _, last = tf.nn.dynamic_rnn(cell=cells, inputs=flow, dtype=tf.float32)
        flow_dim = RNN_units
        flow = last[-1]

    first_fc_output_channels = model_size_info[7]

    first_fc_weights = tf.get_variable('fcw', shape=[flow_dim,
                                                     first_fc_output_channels],
                                       initializer=tf.contrib.layers.xavier_initializer())

    first_fc_bias = tf.Variable(tf.zeros([first_fc_output_channels]))
    first_fc = tf.nn.relu(tf.matmul(flow, first_fc_weights) + first_fc_bias)
    if is_training:
        final_fc_input = tf.nn.dropout(first_fc, dropout_prob)
    else:
        final_fc_input = first_fc

    label_count = model_settings['label_count']

    final_fc_weights = tf.Variable(
        tf.truncated_normal(
            [first_fc_output_channels, label_count], stddev=0.01))

    final_fc_bias = tf.Variable(tf.zeros([label_count]))
    final_fc = tf.matmul(final_fc_input, final_fc_weights) + final_fc_bias
    if is_training:
        return final_fc, dropout_prob
    else:
        return final_fc