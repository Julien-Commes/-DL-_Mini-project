from imgs_to_vid import slice_video, write_video
from naive_model import Mod, train, img2tens, tens2img
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse

def parse_opt():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True, help='video file path')
    return parser.parse_args()


def main(opt):
    """Main function."""
    source = 'video_test.mp4'
    frames, frame_width, frame_height, fps, fourcc = slice_video(source)
    frames = frames[:7]
    X_frames_tensor, y_frames_tensor = img2tens(frames)
    
    mod = Mod(3,3)
    nepochs = 10
    train(mod,X_frames_tensor, y_frames_tensor, nepochs)
    
    output = mod(X_frames_tensor)
    
    scaled_output = tens2img(output)
    
    new_frames = []
    for k in range(len(frames)):
        if k%2 == 0:
            new_frames.append(frames[k])
        else :
            new_frames.append(scaled_output[k//2])
            
    write_video(source, new_frames, frame_width, frame_height, fps, fourcc)

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)