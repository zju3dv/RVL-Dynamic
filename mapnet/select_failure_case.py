from common.Analyser import failure_case_selector
from config import MapNetTestConfig
import os.path as osp

if __name__ == "__main__":
    # input_file = 'mapnet/results/7Scenes_chess_PoseNet.pkl'
    # output_dir = 'mapnet/results/'
    configuration = MapNetTestConfig()
    input_file = osp.join(configuration.figure_output_dir, "result.pkl")
    output_dir = configuration.loss_result_dir
    selector = failure_case_selector(
        configuration,
        input_file,
        output_dir,
        stereo=False,
        t_tolerance=0.10,
        q_tolerance=4.42,
        step=10
    )
    selector.run()
