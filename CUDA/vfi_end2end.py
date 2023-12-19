from imgs_to_vid import slice_video, write_video
from naive_model import Mod, train, img2tens, tens2img
import torch
import argparse

def parse_opt():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=str, help='video file path')
    opt = parser.parse_args()
    return opt


def main(opt):
    """Main function."""
    source = opt.source
    frames, frame_width, frame_height, fps, fourcc = slice_video(source)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    batch_size = 50
    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = img2tens(frames, mode='train', test_size=0.3)    
    trainds = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    trainloader = torch.utils.data.DataLoader(trainds, batch_size=batch_size, shuffle=False)
    testds = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
    testloader = torch.utils.data.DataLoader(testds, batch_size=batch_size, shuffle=False)
    
    mod = Mod(3,3)
    mod = mod.to(device)
    
    nepochs = 50
    train(mod,trainloader, testloader, nepochs)
    
    Frames_tensor = img2tens(frames, mode = 'forward')
    forwardds = torch.utils.data.TensorDataset(Frames_tensor)
    forwardloader = torch.utils.data.DataLoader(forwardds, batch_size=batch_size, shuffle=False)
    
    new_frames = []
    i=0
    for data in forwardloader :
        inputs = data[0]
        output = mod(inputs)
        brightness_factor = 1.05
        output = torch.clamp(output * brightness_factor, 0., 255.)
        scaled_output = tens2img(output)
        for k in range(len(inputs) + len(scaled_output)):
            if k%2 == 0:
                new_frames.append(frames[i*batch_size+k//2])
            else :
                new_frames.append(scaled_output[k//2])
        i+=1
            
    write_video(source, new_frames, frame_width, frame_height, 2*fps, fourcc)

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)