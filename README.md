## Webpage-summarizer:
- **Description:** 
Chrome extension delivers concise and accurate summaries of webpages. Whether you're reading articles, research papers, or long reports, this extension helps you save time by providing the key points at a glance.

![image](https://github.com/user-attachments/assets/9cf670a2-8ca8-407b-b972-7babb28324c9)

## Verified Environment:
[Intel® Tiber™ AI Cloud](https://www.intel.com/content/www/us/en/developer/tools/devcloud/services.html)

[Intel® Data Center GPU Max Series](https://www.intel.com/content/www/us/en/products/details/discrete-gpus/data-center-gpu/max-series.html)

### Environment Setup:
The following information outlines the specifications used for this project:

| Name      | Details                   |
|-----------|---------------------------|
| Platform  | GPU                       |
| Version   | 2.1.40+xpu                |
| OS        | Linux                     |
| Package   | pip                       |


### Environment Setup

```bash
python3 -m venv vllama
source vllama/bin/activate
python -m ipykernel install --user --name vllama
```


Install the packages with help of requirements.txt file:

```
cd video-analytics
pip install -r requirements.txt
```
```
pip install --upgrade --upgrade-strategy eager "optimum[neural-compressor]"
```
Install [IPEX](https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=gpu&version=v2.1.30%2bxpu&os=linux%2fwsl2&package=pip) with the below commands:
```
python -m pip install torch==2.1.0.post3 torchvision==0.16.0.post3 torchaudio==2.1.0.post3 intel-extension-for-pytorch==2.1.40+xpu oneccl_bind_pt==2.1.400+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```
```
python -m pip install numpy==1.26.4
```









