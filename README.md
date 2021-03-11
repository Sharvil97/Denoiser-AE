# Document  Denoiser

Denoiser is required to clean out the received faxed data on in general any text dcoument recieved which is corrupted due to transmission noise. We have developed the following denoisers in order to clean the documents. The denoisers will reduce the noise in the documents which can be further processed using OCR software for downstream NLP tasks.

This repository contains the following denoisers:
1. Simple AE (Auto encoder)
2. DCGAN
3. CycleGAN


To the this repository you need to have an Microsoft Azure account, we need high compute in order to process the high number of medical faxes that we recieved you are free to utilize small datasets like [Kaggle Denoising Dirty Documents Dataset](https://www.kaggle.com/c/denoising-dirty-documents).


For this case we have utilized our custom EHR dataset containing faxed files.

If you are running the code locally please comment out the azureml workspace and run context from train_dcgan.py / train_cyclegan.py files.

To run the code pull the repository using the following command:

```
git clone https://github.com/Sharvil97/Denoiser-AE.git
```

Please install the requisite requirements using the following commands:

```
pip install -r requirements.txt
```

To run the files on the Azure ML:

```
python train_cyclegan.py
```
