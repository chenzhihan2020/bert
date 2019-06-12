# BERT
Implementation of our ideas:
1. Paragraph selection (Answer-based filtering & TF-IDF cosine similarity):
convert_triviaqa_to_squad_format/my_convert_to_squad_format.py
2. Data Augmentation: \
old_versions/run_squad_v1.1.py
3. Handling multiple answers: \
old_versions/run_squad_v3.0.py
4. Combining predictions: \
old_versions/run_squad_test_v3.0.py


Dataset:
TriviaQA: \
Download the dataset: http://nlp.cs.washington.edu/triviaqa/

Running command:
1. Convert the triviaqa data to the squad format: \
python version: 3.7 \
enter into convert_triviaqa_to_squad_format file \
Run the following command. Replace the information during <need to be replaced>, and remove the <>: \
python -m my_convert_to_squad_format \
--triviaqa_file <path to file you want to convert: e.g qa/wikipedia-train.json> \
--squad_file <path to file you want to save the converted file: e.g converted_wikipedia-train.json> \
--wikipedia_dir <path to dataset: e.g ../evidence/wikipedia/>



2. Training the model \
python version: 2.7 \
Replace the information during <need to be replaced>, and remove the <>: \
python run_squad.py \
--vocab_file=<path to the vocab file: e.g gs://tpu0/bert/uncased_L-12_H-768_A-12/vocab.txt> \
--bert_config_file=<path to the vocab file: e.g gs://tpu0/bert/uncased_L-12_H-768_A-12/bert_config.json> \
--init_checkpoint=<path to the pre-trained model: e.g gs://tpu0/bert/uncased_L-12_H-768_A-12/bert_model.ckpt>  \
--do_train=True \
--train_file=<train input file: e.g triviaqa/wikipedia-train_converted.json> \
--do_predict=False \
--predict_file=<test input file: e.g triviaqa/wikipedia-test.json>  \
--train_batch_size=16 \
--learning_rate=3e-5 \
--num_train_epochs=2.0 \
--max_seq_length=384 \
--doc_stride=128 \
--output_dir='<path to directory you want to output: e.g gs://tpu-course-bucket/tpu0/bert/triviaqa/models/bert_base_v3.0_512'> \
--version_2_with_negative=False \
--use_tpu=True \
--tpu_name=<TPU name, e.g haihua-sysu>



3. Testing (Prediction): \
python version: 2.7 \
Replace the information during <need to be replaced>, and remove the <> \
python run_squad.py \
--vocab_file=<path to the vocab file: e.g gs://tpu0/bert/uncased_L-12_H-768_A-12/vocab.txt> \
--bert_config_file=<path to the vocab file: e.g gs://tpu0/bert/uncased_L-12_H-768_A-12/bert_config.json> \
--init_checkpoint=<path to the pre-trained model: e.g gs://tpu0/bert/uncased_L-12_H-768_A-12/bert_model.ckpt>  \
--do_train=False \
--train_file=<train input file: e.g triviaqa/wikipedia-train_converted.json> \
--do_predict=True \
--predict_file=<test input file: e.g triviaqa/wikipedia-test.json>  \
--train_batch_size=16 \
--learning_rate=3e-5 \
--num_train_epochs=2.0 \
--max_seq_length=384 \
--doc_stride=128 \
--output_dir='<path to directory you want to output: e.g gs://tpu-course-bucket/tpu0/bert/triviaqa/models/bert_base_v3.0_512'> \
--version_2_with_negative=False \
--use_tpu=True \
--tpu_name=<TPU name, e.g haihua-sysu>

