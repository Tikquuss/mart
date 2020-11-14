import os, json

config_dic = {
    "hidden_size" : [int, 768],
    "intermediate_size" : [int, 768],
    "vocab_size" : [int, None],
    "word_vec_size" : [int, 300],
    "video_feature_size" : [int, 3072],
    "max_v_len" : [int, 100], 
    "max_t_len" : [int, 25],
    "max_n_sen" : [int, 6],
    "n_memory_cells" : [int, 1],
    "type_vocab_size" : [int, 2],
    "layer_norm_eps" : [float, 1e-12],
    "hidden_dropout_prob" : [float, 0.1],
    "num_hidden_layers" : [int, 2], 
    "attention_probs_dropout_prob" : [float, 0.1],
    "num_attention_heads" : [int, 12],
    "memory_dropout_prob" : [float, 0.1],
    "initializer_range" : [float, 0.02],
    "glove_path" : [str, None], 
    "freeze_glove" : [bool, False],
    "share_wd_cls_weight" : [bool, False],

    "recurrent" : [bool, False],
    "untied" : [bool, False],
    "xl" : [bool, False],
    "xl_grad" : [bool, False],
    "mtrans" : [bool, False],

    "lr" : [float, 1e-4],
    "lr_warmup_proportion" : [float, 0.1], 
    "grad_clip" : [float, 1], 
    "ema_decay" : [float, 0.9999],

    "data_dir" : [str, ""],
    "video_feature_dir" : [str, ""],
    "v_duration_file" : [str, ""],
    "word2idx_path" : [str,  ""],
    "label_smoothing" : [float,  0.1],
    "n_epoch" : [int, 50], 
    "max_es_cnt" : [int,  10],
    "batch_size" : [int, 16], 
    "val_batch_size" : [int, 50],

    "use_beam" : [bool, False],
    "beam_size" : [int, 2], 
    "n_best" : [int, 1], 

    "no_pin_memory" : [bool, False],
    "-num_workers" : [int, 8],
    "exp_id": [str, "res"], 
    "res_root_dir" : [str, "results"], 
    "save_model" : [str, "model"],
    "save_mode": [str, "best"],
    "no_cuda" : [bool,  False],
    "seed" : [int,  2019],
    "debug" : [bool, False],
    "eval_tool_dir" : [str, "./densevid_eval"]
}


def config_file(args) :
    if os.path.isfile(args.config_file):
        with open(args.config_file) as json_data:
            data_dict = json.load(json_data)
            for key, value in data_dict.items():
                conf = config_dic[key]   
                if value == "False":
                    value = False
                elif value == "True" :
                    value = True
                """
                try :
                    setattr(args, key, conf[0](value))
                except :
                    setattr(args, key, value)
                """
                # Allow to overwrite the parameters of the json configuration file.
                try :
                    value = conf[0](value)
                except :
                    pass
                
                if getattr(args, key, conf[1]) == conf[1] :
                    setattr(args, key, value)
    return args


