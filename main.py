import argparse
import nerf_unittest

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pytorch nerf implementation from scratch')
    parser.add_argument('mode', help='execution mode - train/validate/render')
    parser.add_argument('-c', '--config', help='configuration filepath')
    parser.add_argument('-i', '--input', help='input data dirpath')
    parser.add_argument('-o', '--output', help='output data dirpath')

    args = parser.parse_args()

    print('args:', args)

    if args.mode == 'train':
        print('train nerf')
