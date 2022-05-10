# https://github.com/mit-han-lab/once-for-all/blob/4451593507b0f48a7854763adfe7785705abdd78/ofa/imagenet_classification/elastic_nn/training/progressive_shrinking.py

def train(run_manager, args, validate_func=None):
    distributed = isinstance(run_manager, DistributedRunManager)
    if validate_func is None:
        validate_func = validate

    for epoch in range(
        run_manager.start_epoch, run_manager.run_config.n_epochs + args.warmup_epochs
    ):
        train_loss, (train_top1, train_top5) = train_one_epoch(
            run_manager, args, epoch, args.warmup_epochs, args.warmup_lr
        )

        if (epoch + 1) % args.validation_frequency == 0:
            val_loss, val_acc, val_acc5, _val_log = validate_func(
                run_manager, epoch=epoch, is_test=False
            )
            # best_acc
            is_best = val_acc > run_manager.best_acc
            run_manager.best_acc = max(run_manager.best_acc, val_acc)
            if not distributed or run_manager.is_root:
                val_log = (
                    "Valid [{0}/{1}] loss={2:.3f}, top-1={3:.3f} ({4:.3f})".format(
                        epoch + 1 - args.warmup_epochs,
                        run_manager.run_config.n_epochs,
                        val_loss,
                        val_acc,
                        run_manager.best_acc,
                    )
                )
                val_log += ", Train top-1 {top1:.3f}, Train loss {loss:.3f}\t".format(
                    top1=train_top1, loss=train_loss
                )
                val_log += _val_log
                run_manager.write_log(val_log, "valid", should_print=False)

                run_manager.save_model(
                    {
                        "epoch": epoch,
                        "best_acc": run_manager.best_acc,
                        "optimizer": run_manager.optimizer.state_dict(),
                        "state_dict": run_manager.network.state_dict(),
                    },
                    is_best=is_best,
                )