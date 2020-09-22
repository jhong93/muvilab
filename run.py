#!/usr/bin/env python3

from annotator import Annotator
import os
import argparse
import cv2

segment_labels = [
    {'name': 'action', 'color': (0, 255, 0)},
    {'name': 'replay', 'color': (255, 0, 0)},
    {'name': 'other', 'color': (0, 0, 255)},
    {'name': 'toss', 'color': (0, 255, 255)},
]

semantic_labels = [
    {'name': 'axel', 'color': (0, 255, 0)},
    {'name': 'jump', 'color': (255, 0, 0)},
    {'name': 'butterfly', 'color': (0, 0, 255)},
    {'name': 'spin', 'color': (0x7b, 0xe4, 0xbc)},

    # Spins
    {'name': 'upright_spin', 'color': (0x88, 0x7b, 0xe4)},
    {'name': 'camel_spin', 'color': (0xe4, 0x7b, 0xa3)},
    {'name': 'sit_spin', 'color': (0xd7, 0xe4, 0x7b)},

    # Jumps
    {'name': 'toe_loop', 'color': (0x28, 0xd0, 0xd7)},
    {'name': 'salchow', 'color': (0x7b, 0xe4, 0xbc)},
    {'name': 'loop', 'color': (0x88, 0x7b, 0xe4)},
    {'name': 'flip', 'color': (0xe4, 0x7b, 0xa3)},
    {'name': 'lutz', 'color': (0xd7, 0xe4, 0x7b)}
]

parser = argparse.ArgumentParser()
parser.add_argument('video')
parser.add_argument('out_file')
parser.add_argument('-cl', type=float, default=5)
parser.add_argument('-l', choices=['semantic', 'segment'], default='segment')
args = parser.parse_args()

video_file = args.video
video_basename = os.path.basename(video_file)
video_name, ext = os.path.split(video_basename)
out_file = args.out_file

if os.path.isdir(video_file):
    clip_dir = video_file
else:
    clip_dir = 'clips-' + video_basename

annotator = Annotator(
    segment_labels if args.l == 'shot' else semantic_labels,
    clip_dir, annotation_file=out_file, N_show_approx=100)

if not os.path.exists(clip_dir):
    vc = cv2.VideoCapture(video_file)
    fps = vc.get(cv2.CAP_PROP_FPS)
    vc.release()
    clip_frames = round(fps * args.cl)
    print('Clips contain {} frames'.format(clip_frames))
    os.makedirs(clip_dir)
    with open(os.path.join(clip_dir, 'frame_count.txt'), 'w') as fp:
        fp.write(str(clip_frames))
    annotator.video_to_clips(
        video_file, clip_dir, clip_length=clip_frames, overlap=0,
        resize=0.1)
annotator.main()
