from feature_extraction.dataset import AudioProcessor


class LPCProcessor(AudioProcessor):
    # Prepare data

    def __init__(self, training_parameters, data_processing_parameters):
        super().__init__(training_parameters, data_processing_parameters)