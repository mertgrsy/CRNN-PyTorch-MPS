# CRNN-PyTorch-MPS
 Convolutional Recurrent Neural Network (CRNN) for image-based sequence recognition. 
 With this code you can use MPS and CUDA.

 Run demo

  ```sh
  python demo.py -m path/to/model -i data/demo.jpg
  ```

## Train your model

  ```sh
  python train.py -train path/to/train/data -val path/to/val/data
  ```
## Data Format
 [main] \
 PLATE=label \
 YOLO_PLT_3CLASS_1=label X Y w h 
### Example Data
 [main] \
 PLATE=34ANE534 \
 YOLO_PLT_3CLASS_1=0 0.5 0.5 0.3 0.1
 
