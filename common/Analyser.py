import os
import os.path as osp
import cPickle
import numpy as np
from common.pose_utils import quaternion_angular_error
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class failure_case_selector:
    def __init__(self, config, input_file, output_dir, t_tolerance=0.3, q_tolerance=5, stereo=True, model_name='PoseNet', step=1):
        self.config = config
        self.input_file = input_file
        if not osp.isfile(self.input_file):
            print("{:s} is not a file".format(input_file))
            raise EnvironmentError
        self.output_dir = output_dir
        if not osp.isdir(self.output_dir):
            print("{:s} is not a dir".format(output_dir))
            raise EnvironmentError
        self.t_tolerance = t_tolerance
        self.q_tolerance = q_tolerance
        self.stereo = stereo
        self.model_name = model_name
        self.step = step

    def run(self):
        with open(self.input_file, 'rb') as f:
            data = cPickle.load(f)

        t_criterion = lambda t_pred, t_targ: np.linalg.norm(t_pred - t_targ)
        q_criterion = quaternion_angular_error

        pred_poses = data['pred_poses']
        targ_poses = data['targ_poses']
        t_loss = np.asarray([
            t_criterion(p, t) for p, t in zip(pred_poses[:, :3],
                                              targ_poses[:, :3])
        ])
        q_loss = np.asarray([
            q_criterion(p, t) for p, t in zip(pred_poses[:, 3:],
                                              targ_poses[:, 3:])
        ])
        print('Error in translation: median {:3.2f} m,  mean {:3.2f} m\n' \
              'Error in rotation: median {:3.2f} degrees, mean {:3.2f} degree'.format(np.median(t_loss), np.mean(t_loss),
                                                                                      np.median(q_loss), np.mean(q_loss)))

        fig = plt.figure()
        if self.stereo:
            ax = fig.add_subplot(121, projection='3d')
        else:
            ax = fig.add_subplot(121)
        bx = fig.add_subplot(122)
        # plt.subplots_adjust(left=0, bottom=0, right=1, top=1)

        colors = matplotlib.cm.rainbow(np.linspace(0, 1, 6))
        step = self.step
        x = np.vstack((pred_poses[::step, 0].T, targ_poses[::step, 0].T))
        y = np.vstack((pred_poses[::step, 1].T, targ_poses[::step, 1].T))
        pred_colors = [colors[0] for _ in range(len(t_loss))]
        targ_colors = [colors[1] for _ in range(len(t_loss))]
        indices = [i for i in range((len(t_loss)))]
        edge_color = colors[2]
        tq_failure_color = colors[3]
        q_failure_color = colors[4]
        t_failure_color = colors[5]
        t_failure_list = []
        q_failure_list = []
        tq_failure_list = []
        all_loss_list = []
        for i in range(len(t_loss)):
            all_loss_list.append([i, t_loss[i], q_loss[i]])
            if q_loss[i] > self.q_tolerance:
                pred_colors[i] = targ_colors[i] = q_failure_color
                q_failure_list.append([i, q_loss[i]])
            if t_loss[i] > self.t_tolerance:
                pred_colors[i] = targ_colors[i] = t_failure_color
                t_failure_list.append([i, t_loss[i]])
            if q_loss[i] > self.q_tolerance and t_loss[i] > self.t_tolerance:
                pred_colors[i] = targ_colors[i] = tq_failure_color
                tq_failure_list.append([i, t_loss[i], q_loss[i]])

        indices = np.asarray([idx for idx in range(len(t_loss))])
        bx.plot(indices, t_loss, color=edge_color, linewidth=1, label='t_loss')

        if self.stereo:
            ax.plot([0.0, 0.0], [0.0, 0.0], zs=[0.0, 0.0], color=edge_color, linewidth=0.5, label='Success Case')
            ax.plot([0.0, 0.0], [0.0, 0.0], zs=[0.0, 0.0], color=t_failure_color, linewidth=0.5, label='T Failure Case')
            ax.plot([0.0, 0.0], [0.0, 0.0], zs=[0.0, 0.0], color=q_failure_color, linewidth=0.5, label='Q Failure Case')
            ax.plot([0.0, 0.0], [0.0, 0.0], zs=[0.0, 0.0], color=tq_failure_color, linewidth=0.5, label='TQ Failure Case')
            z = np.vstack((pred_poses[::step, 2].T, targ_poses[::step, 2].T))
            for xx, yy, zz, idx in zip(x.T, y.T, z.T, indices):
                if np.array_equal(pred_colors[idx], colors[0]):
                    ax.plot(xx, yy, zs=zz, color=edge_color, linewidth=0.5)
                else:
                    ax.plot(xx, yy, zs=zz, color=pred_colors[idx], linewidth=0.5)
                # self.ax.plot(xx, yy, zs=zz, color=edge_color)
            ax.scatter(x[0, :], y[0, :], zs=z[0, :], color=pred_colors, depthshade=0, s=0.8)
            ax.scatter(x[1, :], y[1, :], zs=z[1, :], color=targ_colors, depthshade=0, s=0.8)
            ax.view_init(azim=119, elev=13)
        else:
            ax.plot(x, y, color=edge_color, linewidth=0.5)
            ax.scatter(x[0, :], y[0, :], color=pred_colors, s=0.8)
            ax.scatter(x[1, :], y[1, :], color=targ_colors, s=0.8)

        # plt.show(block=True)
        legend = ax.legend(
            loc='upper center',
            shadow=True,
            fontsize='x-small'
        )
        if self.config.display:
            plt.show(block=True)

        image_filename = osp.join(osp.expanduser(self.output_dir),
                                  '{:s}_fig.png'.format(self.model_name))
        fig.savefig(image_filename)
        t_failure_filename = osp.join(osp.expanduser(self.output_dir),
                                      't_failure_list.txt')
        with open(t_failure_filename, 'w') as f:
            for item in t_failure_list:
                f.write("{:05d}: {}\n".format(item[0], item[1]))
        q_failure_filename = osp.join(osp.expanduser(self.output_dir),
                                      'q_failure_list.txt')
        with open(q_failure_filename, 'w') as f:
            for item in q_failure_list:
                f.write("{:05d}: {}\n".format(item[0], item[1]))
        tq_failure_filename = osp.join(osp.expanduser(self.output_dir),
                                       'tq_failure_list.txt')
        with open(tq_failure_filename, 'w') as f:
            for item in tq_failure_list:
                f.write("{:05d}: t_loss: {}, q_loss: {}\n".format(item[0], item[1], item[2]))

        all_loss_filename = osp.join(osp.expanduser(self.output_dir),
                                     'all_loss_list.txt')
        with open(all_loss_filename, 'w') as f:
            for item in all_loss_list:
                f.write("{:05d}: t_loss: {}, q_loss: {}\n".format(item[0], item[1], item[2]))
