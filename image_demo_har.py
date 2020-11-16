import cv2
import time
import argparse
import os
import torch
import pickle
from PIL import Image

import posenet


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--scale_factor', type=float, default=1.0)
parser.add_argument('--notxt', action='store_true')
parser.add_argument('--image_dir', type=str, default='./images')
parser.add_argument('--output_dir', type=str, default='./output')
args = parser.parse_args()


def main():
    model = posenet.load_model(args.model)
    model = model.cuda()
    output_stride = model.output_stride

    if args.output_dir:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    # solo lee imagenes .png o .jpg
    filenames = [
        f.path for f in sorted(os.scandir(args.image_dir), 
          key=lambda e: e.name) if f.is_file() and f.path.endswith(('.png', '.jpg'))]

    tt = {}
    start = time.time()
    for f in filenames:
        input_image, draw_image, output_scale = posenet.read_imgfile(
            f, scale_factor=args.scale_factor, output_stride=output_stride)

        with torch.no_grad():
            input_image = torch.Tensor(input_image).cuda()

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
                heatmaps_result.squeeze(0),
                offsets_result.squeeze(0),
                displacement_fwd_result.squeeze(0),
                displacement_bwd_result.squeeze(0),
                output_stride=output_stride,
                max_pose_detections=2,
                min_pose_score=0.55)

        keypoint_coords *= output_scale
        '''if args.output_dir:
            draw_image = posenet.draw_skel_and_kp(
                draw_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.25, min_part_score=0.25)

            cv2.imwrite(os.path.join(args.output_dir, os.path.relpath(f, args.image_dir)), draw_image)'''

        if not args.notxt:
            print("Results for image: %s" % f)
            
            pp = []
            for pi in range(len(pose_scores)):
                if pose_scores[pi] <= 0.55:
                    break
                print('Pose #%d, score = %f' % (pi, pose_scores[pi]))
                for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
                    print('Keypoint %s, score = %f, coord = %s' % (posenet.PART_NAMES[ki], s, c))

                pp.append(keypoint_coords[pi,:,:])
        if pp:
            ff = {f: pp}
            tt.update(ff)

    name ='./data/' + str(args.image_dir.split('/')[-1]) + '.pickle'
    filename = open(name, "wb")

    pickle.dump(tt, filename)

    # pickfile = open('datos.txt', 'w')
    # pickfile.write(str(tt))

    print('Average FPS:', len(filenames) / (time.time() - start))


if __name__ == "__main__":
    main()
