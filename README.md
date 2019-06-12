# BERT
Implementation of our ideas:
1. Paragraph selection (Answer-based filtering & TF-IDF cosine similarity):
convert_triviaqa_to_squad_format/my_convert_to_squad_format.py
2. Data Augmentation:
old_versions/run_squad_v1.1.py
3. Handling multiple answers:
old_versions/run_squad_v3.0.py
4. Combining predictions:
old_versions/run_squad_test_v3.0.py


Running command:
1. TriviaQA: \
Download the dataset: http://nlp.cs.washington.edu/triviaqa/

2. Convert the triviaqa data to the squad format: \
python version: 3.7 \
enter into convert_triviaqa_to_squad_format file \
Run the following command. Replace the information during <need to be replaced>, and remove the <>: \
python -m my_convert_to_squad_format \
--triviaqa_file <path to file you want to convert: i.g. qa/wikipedia-train.json> \
--squad_file <path to file you want to save the converted file: i.g. converted_wikipedia-train.json> \
--wikipedia_dir <path to dataset: i.g. ../evidence/wikipedia/>



3. Run the bert model:
python version: 2.7
Run the following command for training the model:
python run_squad.py \
--vocab_file=<path to the vocab file: i.g. gs://tpu0/bert/uncased_L-12_H-768_A-12/vocab.txt> \
--bert_config_file=<path to the vocab file: i.g. gs://tpu0/bert/uncased_L-12_H-768_A-12/bert_config.json> \
--init_checkpoint=<path to the pre-trained model: i.g. gs://tpu0/bert/uncased_L-12_H-768_A-12/bert_model.ckpt>  \
--do_train=True \
--train_file=<train input file: i.g. triviaqa/wikipedia-train_converted.json> \
--do_predict=False \
--predict_file=<test input file: i.g. triviaqa/wikipedia-test.json>  \
--train_batch_size=16 \
--learning_rate=3e-5 \
--num_train_epochs=2.0 \
--max_seq_length=384 \
--doc_stride=128 \
--output_dir='<path to directory you want to output: i.g. gs://tpu-course-bucket/tpu0/bert/triviaqa/models/bert_base_v3.0_512'> \
--version_2_with_negative=False \
--use_tpu=True \
--tpu_name=<TPU name, i.g. haihua-sysu>

Replace the information during <need to be replaced>, and remove the <>

Run the following command for testing (Prediction):
python run_squad.py \
--vocab_file=<path to the vocab file: i.g. gs://tpu0/bert/uncased_L-12_H-768_A-12/vocab.txt> \
--bert_config_file=<path to the vocab file: i.g. gs://tpu0/bert/uncased_L-12_H-768_A-12/bert_config.json> \
--init_checkpoint=<path to the pre-trained model: i.g. gs://tpu0/bert/uncased_L-12_H-768_A-12/bert_model.ckpt>  \
--do_train=False \
--train_file=<train input file: i.g. triviaqa/wikipedia-train_converted.json> \
--do_predict=True \
--predict_file=<test input file: i.g. triviaqa/wikipedia-test.json>  \
--train_batch_size=16 \
--learning_rate=3e-5 \
--num_train_epochs=2.0 \
--max_seq_length=384 \
--doc_stride=128 \
--output_dir='<path to directory you want to output: i.g. gs://tpu-course-bucket/tpu0/bert/triviaqa/models/bert_base_v3.0_512'> \
--version_2_with_negative=False \
--use_tpu=True \
--tpu_name=<TPU name, i.g. haihua-sysu>

Replace the information during <need to be replaced>, and remove the <>
