import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices("GPU")

tf.config.experimental.set_memory_growth(physical_devices[0], True)

import yaml
import numpy as np

# from ptycle_net.ptycle.cyclegan import CycleGAN
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.callbacks import *

from ptynet.models import *
from ptynet.losses import *
import random
import argparse
import time


def set_seed(seed=2134):
    # tf.keras.utils.set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # # When running on the CuDNN backend, two further options must be set
    # os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    # os.environ["TF_DETERMINISTIC_OPS"] = "1"
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def main(args):
    set_seed(0)

    config = yaml.safe_load(open(args.dataset))

    config["hyper"]["dist"] = args.dist
    config["hyper"]["n_refine"] = args.n_refine
    config["hyper"]["probe_mode"] = args.probe_mode

    config["model"]["mode"] = args.mode

    config["hyper"]["save_path"] += "_{}_r{}_{}".format(args.mode, args.n_refine, args.probe_mode)

    if not args.dist:
        config["hyper"]["save_path"] += "_mse"

    if args.mode == "3d":
        print("Load model PtySD3Net")
        unsp_model = PtySD3Net(config, args.pretrained)
        unsp_model.model.summary()
    elif args.mode == "2d":
        unsp_model = PtyBase2D(config, args.pretrained)
        unsp_model.model.summary()
    elif args.mode == "autonn":
        print("Load model AutoPhaseNN")
        unsp_model = AutoPhaseNN(config, args.pretrained)
        unsp_model.model.summary()
    elif args.mode == "ptychonn":
        print("Load model PtychoNN")
        unsp_model = PtychoNN(config, args.pretrained)
        unsp_model.model.summary()
    else:
        print("Not available options: 3d or 2d model")

    unsp_model.create_dataset()

    start = time.time()
    hist = unsp_model.train(args.epoch)
    print("Total training time: ", time.time() - start)

    np.save(config["hyper"]["save_path"] + "/hist_train.npy", hist.history)

    print("Load trained model and inference: ")
    unsp_model.inference()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("dataset", type=str, help="Path to dataset configs")

    parser.add_argument("--mode", type=str, default="3d", help="Model type 3D or 2D (optional)")
    parser.add_argument("--n_refine", type=int, default=5, help="Refinement step for enhance reconstruction")
    parser.add_argument(
        "--probe_mode", type=str, default="multi_c", help="Refine probe mode: single, single_c, multi, multi_c"
    )

    parser.add_argument("--pretrained", type=str, default="", help="Path to pretrained model (optional)")
    parser.add_argument("--dist", type=bool, default=False, help="Using Poisson distribution for output")
    parser.add_argument("--epoch", type=int, default=20, help="Training epochs")

    args = parser.parse_args()
    main(args)
