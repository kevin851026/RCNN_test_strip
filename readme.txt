strip_unused.py �Ψ��Y��w�ᵲ��model(.pb��)  
python strip_unused.py --input_graph=frozen_model.pb --output_graph=new.pb --input_node_names="Placeholder" --output_node_names="cnn_1/Softmax" --input_binary=true
�����O�]�i�ΨӽT�w��J�P��X�I�O�_�s�bmodel��

frozen.py �Ψӭᵲmodel
�ѼƬ�
--model_dir MODEL_DIR
                        Model folder to export
--output_node_names OUTPUT_NODE_NAMES
                        The name of the output nodes, comma separated.

python frozen.py --model_dir="300/" --output_node_names="cnn_1/dense_3/BiasAdd"

python strip_unused.py --input_graph=cnn_frozen_model.pb --output_graph=new.pb --input_node_names="Placeholder" --output_node_names="cnn_1/dense_3" --input_binary=true