import os
import argparse

class CLogger:
    def output_args(self,args,output_file,bool_print):
        arg_dict = args.__dict__
        if bool_print:
            for k,v in arg_dict.items():
                print('{key}:{value}'.format(key=k,value=v))
        if not os.path.isfile(output_file):
            fid = open(output_file, 'w')
            fid.close()

        with open(output_file, 'w') as f:
            f.writelines('------------------ start ------------------' + '\n')
            for eachArg, value in arg_dict.items():
                f.writelines(eachArg + ' : ' + str(value) + '\n')
            f.writelines('------------------- end -------------------')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--map_path', default='/media/alv-2/Rambo/Data/KITTI-Map/map_0.08/')
    parser.add_argument('--patch_path', default='/media/alv-2/Rambo/Data/KITTI-patches/')
    parser.add_argument('--seq_id', default='51')
    parser.add_argument('--patch_width_x', type=int,default=5)
    parser.add_argument('--patch_length_y', type=int, default=5)
    parser.add_argument('--patch_height_z', type=float, default=1.0)
    parser.add_argument('--patch_radius', type=float, default=6.0)
    parser.add_argument('--patch_mini_points', type=int, default=500)
    FLAGS = parser.parse_args()

    output_file = "./args.txt"
    logger = CLogger()
    logger.output_args(FLAGS,output_file,bool_print=True)


