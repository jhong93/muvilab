#!/usr/bin/env python3

from annotator import Annotator
import os
import argparse
import cv2

labels = [
    {'name': 'action', 'color': (0, 255, 0)},
    {'name': 'replay', 'color': (255, 0, 0)},
    {'name': 'other', 'color': (0, 0, 255)}
]

parser = argparse.ArgumentParser()
parser.add_argument('video')
parser.add_argument('out_file')
parser.add_argument('-cl', type=float, default=5)
args = parser.parse_args()


video_file = args.video
video_basename = os.path.basename(video_file)
video_name, ext = os.path.split(video_basename)
out_file = args.out_file
clip_dir = 'clips-' + video_basename

vc = cv2.VideoCapture(video_file)
fps = vc.get(cv2.CAP_PROP_FPS)
vc.release()
clip_frames = round(fps * args.cl)
print('Clips contain {} frames'.format(clip_frames))

annotator = Annotator(
    labels, clip_dir, annotation_file=out_file, N_show_approx=100)

if not os.path.exists(clip_dir):
    os.makedirs(clip_dir)
    annotator.video_to_clips(
        video_file, clip_dir, clip_length=clip_frames, overlap=0,
        resize=0.1)
annotator.main()
