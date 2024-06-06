# LLM-Diarize-ASR-Agnostic

Repository for code used in "LLM-based speaker diarization correction: A generalizable approach" manuscript.

## 1. Transcribe audio files

First step to reproduce the results is to transcribe the audio files. In this paper we used AWS Transcribe, Azure Speech to Text, Google Speech to Text and WhisperX to transcribe our audio files and thus we offer support for processing transcriptions from these services. But it is also possible to use other transcription services, as long as the output includes the timestamps of the transcribed text at the word level along with speaker identification.

After transcribing the audio files, you should have a csv file with the following columns:

* `utt_id`: unique identifier for the transcription
* `file_path_original`: path to the reference transcription file
* `file_path_asr`: path to the transcription file from the ASR system

## 2. Data Preprocessing

The next step is to preprocess the data. This step includes the following tasks:

* Create completions for the LLM, by transferring speaker labels from the reference transcription to the ASR transcription
* Create the training data by:
  * Filtering the full data to contain only the training data
  * Splitting the prompts-completions into manageable chunks
  * Applying the prompt-completion format to the training data
  * Tokenizing the training data
* Create the evaluation data by:
  * Filtering the full data to contain only the evaluation data
  * Splitting the prompts-completions into manageable chunks
  * Mapping the prompt-completion chunks to the original evaluation data
  * Saving the evaluation data for later use

These steps can be done by following the tutorials in the `preprocess` folder. Namely, the tutorials are:

* `data_preprocess.ipynb`: This notebook shows how to preprocess the data for the LLM and the training data.
* `training_preprocess.ipynb`: This notebook shows how to preprocess the data for the training data.
* `testing_preprocess.ipynb`: This notebook shows how to preprocess the data for the evaluation data.

## 3. Fine-tuning the LLM

There are various ways to fine-tune an LLM on the training data. In this paper we used the `transformers` library from Hugging Face to fine-tune the LLM in AWS SageMaker. The code for fine-tuning the LLM can be found in the `fine_tuning` folder. The tutorial for runing the training can be found there as well as the training script.

## 4. Deploying the LLM and evaluating the results

After fine-tuning the LLM, the next step is to deploy the LLM and evaluate the results. The code for deploying the LLM and evaluating the results can be found in the `evaluation` folder. There exists a tutorial for each of the evaluation steps:

* `deploy_model.ipynb`: This notebook shows how to deploy the LLM on an AWS SageMaker endpoint and how to generate completions for the evaluation data.
* `evaluate_model.ipynb`: This notebook shows how to post-process and evaluate the completions generated by the LLM.

## 5. Merging multiple LLMs

To replicate the results of the paper, you will need to fine-tune multiple LLMs (which use different ASR transcribed data each) and merge them using the merge-kit library. In the paper we used the TIES merging approach, but there are also others to explore with this framework found in GitHub [heres](https://github.com/arcee-ai/mergekit/tree/main#merge-methods). We provide as an example, the YAML configuration file used to merge the LLMs in the `merge` folder. To create the ensemble LLM clone the merge-kit repository and from the root directory run the following command:

```bash
mergekit-yaml path/to/config_ties.yaml \
    path/to/output_dir \
    --copy-tokenizer \
    --out-shard-size 1B \
    --lazy-unpickle
```
