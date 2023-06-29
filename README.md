# Improving nighttime object detection using image-to-image translation

## CycleGAN

Unpaired Image-to-Image Translation using Cycle-consistent adversarial networks(ICCV 2017)

- 이미지로 구성된 서로 다른 도메인(domain)의 데이터셋이 있을 때 X 도메인 데이터셋을 Y 도메인 데이터셋으로 변환(image-to-image translation)
- For many tasks, paired training data will not be available. Capturing special characteristics of one image collection and figuring out how these characteristics could be translated into other image collection, all in the absence of any paired training examples.
- pix2pix와 비교했을 때, 쌍을 이루지 않은 데이터셋(unpaired dataset)으로 학습이 가능한 unpaired image-to-image translation 방법을 제안
- 핵심 아이디어로 cycle-consistent loss를 제안해 다양한 task에서 뛰어난 성능을 보여줌(style transfer, object transfiguration, attirbute transfer, photo enhancement)
- 문제점) adversarial loss만 사용할 경우 생성자는 어떤 입력이든 y 도메인에 해당하는 하나의 이미지만 제시할 수도 있음 (mode collapse : where all input images map to the same output image)→ 추가적인 제약 조건이 필요(cycle-consistency loss 사용)
- cycle-consistency loss : 생성자(G(x))가 다시 원본 이미지(x)로 구성할 수 있도록 translation → 2개의 generator 사용(G, F : G and F should be inverses of each other)
- 2개의 판별자 사용 : Dx aims to distinguish between images {x} and translated images {F(y)}. Dy aims to distinguish between images {y} and translated images {G(x)}.
- 2개의 loss : adversarial loss(1), cycle consistency loss(2)

![스크린샷 2023-03-17 시간: 12.15.45.png](Improving%20nighttime%20object%20detection%20using%20image-t%20be5d82e7e83e41a291f5bbc57c4c8302/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2023-03-17_%25E1%2584%2589%25E1%2585%25B5%25E1%2584%2580%25E1%2585%25A1%25E1%2586%25AB_12.15.45.png)

- 위와 동일하게 generator F에 대해서도 GAN loss가 정의

cycle consistency loss :

![스크린샷 2023-03-17 시간: 12.16.07.png](Improving%20nighttime%20object%20detection%20using%20image-t%20be5d82e7e83e41a291f5bbc57c4c8302/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2023-03-17_%25E1%2584%2589%25E1%2585%25B5%25E1%2584%2580%25E1%2585%25A1%25E1%2586%25AB_12.16.07.png)

objective function :

![스크린샷 2023-03-17 시간: 12.19.41.png](Improving%20nighttime%20object%20detection%20using%20image-t%20be5d82e7e83e41a291f5bbc57c4c8302/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2023-03-17_%25E1%2584%2589%25E1%2585%25B5%25E1%2584%2580%25E1%2585%25A1%25E1%2586%25AB_12.19.41.png)

- residual block을 활용, instance normalization 사용(generator architecture)
- 이미지 내 패치 단위로 진위 여부를 판별하는 판별자 사용(PatchGAN) - we use 70*70 patchGANs which aim to classify whether 70*70 overlapping image patches are real or fake → such a patch level discriminator architecture has fewer parameters than an full-image discriminator
- cross entropy loss에 비해 학습이 안정화되고(gradient vanishing) , 실제 이미지의 분포와 가까운 이미지를 생성하기 위해 MSE 기반의 loss 사용 -  we replace the negative log likelihood objective by a least square loss

![스크린샷 2023-03-24 시간: 14.05.53.png](Improving%20nighttime%20object%20detection%20using%20image-t%20be5d82e7e83e41a291f5bbc57c4c8302/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2023-03-24_%25E1%2584%2589%25E1%2585%25B5%25E1%2584%2580%25E1%2585%25A1%25E1%2586%25AB_14.05.53.png)

- Replay buffer : 생성자가 만든 이전의 50개의 이미지를 저장해 두고, 이를 활용해 판별자를 업데이트 - To reduce model oscillation, we update Dx and Dy using a history of generated images(we keep an image buffer that stores the 50 previously generated images)
- Identity loss : 색상 구성을 보존해야 할 때 사용 가능(그림을 사진으로 변경할 때와 같이 색상 정보가 유지되어야 하는 task에서 유용하게 활용 가능)
- 한계) 모양 정보를 포함한 content의 변경이 필요한 경우나 학습 데이터에 포함되지 않은 사물을 처리하는 경우 한계점을 갖는다.

<training details>

- We train our networks from scratch, with a learning rate of 0.0002.
- We keep the same learning rate for the first 100 epochs and linearly decay the rate to zero over the next 100 epochs.
- Weights are initialized from a Gaussian distribution N (0, 0.02).
- We use 6 residual blocks for 128 × 128 training images, and 9 residual blocks for 256 × 256 or higher-resolution training images.

논문 저자의 github : [https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

## ****BDD100K: A Large-scale Diverse Driving Video Database for autonomous driving(released in 2018)****

이미지 크기 : 1280 * 720 → image resize 512 * 512 / 256 * 256(cycleGAN 논문에서 일반적으로 128 * 128 또는 256 * 256으로 scaling)

총 100,000개의 이미지(label 포함 - timeofday, category, box2d 등)

train : test : val = 7 : 2 : 1

training set 중 daytime image : 36,728개

training set 중 nigtht image : 27,971개

validation set 중 daytime image : 5,258개

validation set 중 night image : 3,929개

![스크린샷 2023-03-23 시간: 21.18.26.png](Improving%20nighttime%20object%20detection%20using%20image-t%20be5d82e7e83e41a291f5bbc57c4c8302/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2023-03-23_%25E1%2584%2589%25E1%2585%25B5%25E1%2584%2580%25E1%2585%25A1%25E1%2586%25AB_21.18.26.png)

## GPU setup

- Sign up KHU seraph
- ssh remote server access
- NAS에 anaconda3 다운로드 및 Pyhon 3.8 가상환경 생성
- nvcc —version / nvidia-smi 로 cuda 버전 확인(cuda = 11.7)
- Install Pytorch, torchvision, torchaudio(GPU version)

```bash
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia 
```

- NAS에 github repository clone / dataset download
- You can build your own dataset by setting up the following directory structure :

```markdown
.
├── datasets                   
|   ├── <dataset_name>         # i.e. day2night
|   |   ├── train              # Training
|   |   |   ├── A              # Contains domain A images (i.e. Daytime)
|   |   |   └── B              # Contains domain B images (i.e. Night)
|   |   └── test               # Testing
|   |   |   ├── A              # Contains domain A images (i.e. Daytime)
|   |   |   └── B              # Contains domain B images (i.e. Night)
```

- srun 명령어로 resource 할당

```bash
srun --gres=gpu:1 --cpus-per-gpu=8 --mem-per-gpu=5G --partition debug_ugrad --pty "$SHELL”
```

- batch job을 위해 sbatch 명령어 실행
    
    ```bash
    sbatch test.sh 
    ```
    
    - VScode SFTP extension install
    - SFTP setting
    - job submit

## Result

input/output : RGB 3 channel image(512 * 512)

batch size = 8(36728 daytime images, 27971 night images), training epoch = 200

![Untitled](Improving%20nighttime%20object%20detection%20using%20image-t%20be5d82e7e83e41a291f5bbc57c4c8302/Untitled.jpeg)

![Untitled](Improving%20nighttime%20object%20detection%20using%20image-t%20be5d82e7e83e41a291f5bbc57c4c8302/Untitled%201.jpeg)

![Untitled](Improving%20nighttime%20object%20detection%20using%20image-t%20be5d82e7e83e41a291f5bbc57c4c8302/Untitled%202.jpeg)

![Untitled](Improving%20nighttime%20object%20detection%20using%20image-t%20be5d82e7e83e41a291f5bbc57c4c8302/Untitled%203.jpeg)

## Detection result(YOLOv5)

1. daytime images 10000장 (train : val : test = 6 : 2 : 2)
2. daytime images(5000장), night images(5000장) 
3. translated daytime images(5000장), translated night images(5000장)

epoch = 300, batch_size = 16, image_size = 512 * 512, early stopping 설정(mAP 기준으로 patience = 100으로 설정) 

test_dataset : night images(2000장)

![Untitled](Improving%20nighttime%20object%20detection%20using%20image-t%20be5d82e7e83e41a291f5bbc57c4c8302/Untitled%204.jpeg)

![Untitled](Improving%20nighttime%20object%20detection%20using%20image-t%20be5d82e7e83e41a291f5bbc57c4c8302/Untitled%205.jpeg)

1. daytime images 

![bicycle : 0, bus : 1, car : 2, motorcycle : 3, pedestrian : 4, rider : 5 , traffic light : 6, traffic sign : 7, train : 8, truck : 9](Improving%20nighttime%20object%20detection%20using%20image-t%20be5d82e7e83e41a291f5bbc57c4c8302/Untitled.png)

bicycle : 0, bus : 1, car : 2, motorcycle : 3, pedestrian : 4, rider : 5 , traffic light : 6, traffic sign : 7, train : 8, truck : 9

1. daytime images + night images 

![bicycle : 0, bus : 1, car : 2, motorcycle : 3, pedestrian : 4, rider : 5 , traffic light : 6, traffic sign : 7, train : 8, truck : 9](Improving%20nighttime%20object%20detection%20using%20image-t%20be5d82e7e83e41a291f5bbc57c4c8302/Untitled%201.png)

bicycle : 0, bus : 1, car : 2, motorcycle : 3, pedestrian : 4, rider : 5 , traffic light : 6, traffic sign : 7, train : 8, truck : 9

1. translated images(daytime images + night images)

![bicycle : 0, bus : 1, car : 2, motorcycle : 3, pedestrian : 4, rider : 5 , traffic light : 6, traffic sign : 7, train : 8, truck : 9](Improving%20nighttime%20object%20detection%20using%20image-t%20be5d82e7e83e41a291f5bbc57c4c8302/Untitled%202.png)

bicycle : 0, bus : 1, car : 2, motorcycle : 3, pedestrian : 4, rider : 5 , traffic light : 6, traffic sign : 7, train : 8, truck : 9