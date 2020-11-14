"""
dset_name=
data_dir=
video_feature_dir=
v_duration_file=
word2idx_path=
max_v_len=
max_t_len=
max_n_sen=
recurrent=
untied=
mtrans=

python main.py --dset_name $dset_name  \
	--data_dir $data_dir \
	--video_feature_dir $video_feature_dir \
	--v_duration_file $v_duration_file \
	--word2idx_path $word2idx_path \
	--max_v_len $max_v_len \
	--max_t_len $max_t_len \
	--max_n_sen $max_n_sen  \
	--recurrent $recurrent  \
	--untied $untied  \
	--mtrans $mtrans  
"""



from src.rtransformer.dataset import RecursiveCaptionDataset as RCDataset

if __name__ == "__main__":
	
	"""parse and preprocess cmd line args"""
    parser = argparse.ArgumentParser()

    parser.add_argument("--dset_name", type=str, default="anet", choices=["anet", "yc2"], help="Name of the dataset, will affect data loader, evaluation, etc")
    parser.add_argument("--data_dir", required=True, help="dir containing the splits data files")
    parser.add_argument("--video_feature_dir", required=True, help="dir containing the video features")
    parser.add_argument("--v_duration_file", required=True, help="filepath to the duration file")
    parser.add_argument("--word2idx_path", type=str, default="./cache/word2idx.json")
    parser.add_argument("--max_v_len", type=int, default=100, help="max length of video feature")
    parser.add_argument("--max_t_len", type=int, default=25, help="max length of text (sentence or paragraph), 30 for anet, 20 for yc2")
    parser.add_argument("--max_n_sen", type=int, default=6, help="for recurrent, max number of sentences, 6 for anet, 10 for yc2")
    parser.add_argument("--recurrent", action="store_true", help="Run recurrent model")
    parser.add_argument("--untied", action="store_true", help="Run untied model")
    parser.add_argument("--mtrans", action="store_true", help="Masked transformer model for single sentence generation")

    opt = parser.parse_args()

    train_dataset = RCDataset(
        dset_name=opt.dset_name,
        data_dir=opt.data_dir, 
        video_feature_dir=opt.video_feature_dir,
        duration_file=opt.v_duration_file,
        word2idx_path=opt.word2idx_path, 
        max_t_len=opt.max_t_len,
        max_v_len=opt.max_v_len, 
        max_n_sen=opt.max_n_sen, 
        mode="train",
        recurrent = opt.recurrent, untied=opt.untied or opt.mtrans)



