# *_*coding:utf-8 *_*
import os

AFFECTGPT_ROOT = './'
EMOTION_WHEEL_ROOT = './emotion_wheel'
OUTSIDE_WHEEL_MAPPING = os.path.join(EMOTION_WHEEL_ROOT, 'wheel_mapping.npz')
RESULT_ROOT = os.path.join(AFFECTGPT_ROOT, 'output/results')

###########################################
## 所有模型的存储路径 [放在一个路径下]
###########################################
PATH_TO_LLM = {
    'Qwen25': 'models/Qwen2.5-7B-Instruct',
}

PATH_TO_VISUAL = {
    'CLIP_VIT_LARGE': 'models/clip-vit-large-patch14',
}

PATH_TO_AUDIO = {
    'HUBERT_LARGE':  'models/chinese-hubert-large',
}

PATH_TO_MLLM = {
    ## For Qwen-Audio
    'qwen-audio-chat':            '../models/qwen-audio-chat',
    ## For SALMONN
    'salmonn_7b':                 '../models/salmonn_7b.pth',
    'vicuna-7b-v1.5':             '../models/vicuna-7b-v1.5',
    'BEATs':                      '../models/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt', 
    'whisper-large-v2':           '../models/whisper-large-v2', 
    ## For Video-ChatGPT
    'video_chatgpt-7B':           '../models/video_chatgpt-7B.bin',
    'LLaVA-7B-Lightening-v1-1':   '../models/LLaVA-7B-Lightening-v1-1',
    'clip-vit-large-patch14':     '../models/clip-vit-large-patch14',
    ## For Video-LLaMA
    'llama-2-7b-chat-hf':         '../models/llama-2-7b-chat-hf',
    'imagebind_huge':             '../models/imagebind_huge.pth',
    'video_llama_vl':             '../models/VL_LLaMA_2_7B_Finetuned.pth',
    'video_llama_al':             '../models/AL_LLaMA_2_7B_Finetuned.pth',
    'blip2_pretrained_flant5xxl': '../models/blip2_pretrained_flant5xxl.pth',
    'bert-base-uncased':          '../models/bert-base-uncased',
    'eva_vit_g':                  '../models/eva_vit_g.pth',
    ## For Chat-UniVi
    'Chat-UniVi':                 '../models/Chat-UniVi',
    ## For LLaMA-VID
    'llama-vid':                  '../models/llama-vid-7b-full-224-video-fps-1',
    ## For mPLUG-Owl
    'mplug-owl':                  '../models/mplug-owl-llama-7b-video',
    ## For Otter
    'otter':                      '../models/OTTER-Video-LLaMA7B-DenseCaption',
    ## For VideoChat
    'vicuna-7b-v0':               '../models/vicuna-7b-v0',
    'videochat_7b':               '../models/videochat_7b.pth',
    ## For VideoChat2
    'umt_l16_qformer':            '../models/umt_l16_qformer.pth',
    'videochat2_7b_stage2':       '../models/videochat2_7b_stage2.pth',
    'videochat2_7b_stage3':       '../models/videochat2_7b_stage3.pth',
    ## For Video-LLaVA
    'Video-LLaVA':                '../models/Video-LLaVA-7B',
}


###################################################
## 所有数据集的存储路径 [所有标签都在 MER2026 路径下]
###################################################
## 数据根目录 (2026-08-25 更新): 真实数据位于 /root/fsas/AffectGPT_dataset
## 注意: 该目录下 audio/ video/ openface_face/ 三个扁平目录是失效的符号链接(指向已删除的
##       /root/fsas/MER26/Dataset), 不要使用! 完整数据按分组存放在 *_7z/{group}/ 下。
DATA_DIR = {
    'MER2026':          '/root/fsas/AffectGPT_dataset',
}
## 各组完整数据 (已核验: 分组内 音频/视频/人脸 覆盖各自 csv 全部样本)
PATH_TO_RAW_AUDIO = {
    'Human':          os.path.join(DATA_DIR['MER2026'], 'audio_7z/audio_track2_train_human/audio'),
    'MERCaptionPlus': os.path.join(DATA_DIR['MER2026'], 'audio_7z/audio_track2_train_mercaptionplus/audio'),
    'MER2026OV':      os.path.join(DATA_DIR['MER2026'], 'audio_7z/audio_track1_track2_candidate/audio'),
}
PATH_TO_RAW_VIDEO = {
    'Human':          os.path.join(DATA_DIR['MER2026'], 'video_7z/video_track2_train_human/video'),
    'MERCaptionPlus': os.path.join(DATA_DIR['MER2026'], 'video_7z/video_track2_train_mercaptionplus/video'),
    'MER2026OV':      os.path.join(DATA_DIR['MER2026'], 'video_7z/video_track1_track2_candidate/video'),
}
PATH_TO_RAW_FACE = {
    'Human':          os.path.join(DATA_DIR['MER2026'], 'openface_7z/openface_track2_train_human/openface_face'),
    'MERCaptionPlus': os.path.join(DATA_DIR['MER2026'], 'openface_7z/openface_track2_train_mercaptionplus/openface_face'),
    'MER2026OV':      os.path.join(DATA_DIR['MER2026'], 'openface_7z/openface_track1_track2_candidate/openface_face'),
}
PATH_TO_TRANSCRIPTIONS = {
    'Human':          os.path.join(DATA_DIR['MER2026'], 'subtitle_chieng.csv'),
    'MERCaptionPlus': os.path.join(DATA_DIR['MER2026'], 'subtitle_chieng.csv'),
    'MER2026OV':      os.path.join(DATA_DIR['MER2026'], 'subtitle_chieng.csv'),
}
PATH_TO_LABEL = {
    ## GRPO 训练用 9:1 划出的 human 90% (1379 条); 全量在 track2_train_human.csv (1532 条)。
    'Human':          os.path.join(DATA_DIR['MER2026'], 'track2_train_human_train90.csv'),
    ## SFT 训练用去重版 (剔除了与 test10 重叠的 51 条, 避免数据泄漏);
    ## 原版全量在 track2_train_mercaptionplus.csv (31,327 条)。
    'MERCaptionPlus': os.path.join(DATA_DIR['MER2026'], 'track2_train_mercaptionplus_dedup.csv'),
    ## 官方测试集 (20000 candidates, 无 openset 标签, 用于最终提交);
    ## 重跑开发期验证时, 改为指向 9:1 划出的 human 10% 测试集:
    ##   os.path.join(DATA_DIR['MER2026'], 'track2_train_human_test10.csv')
    'MER2026OV':      os.path.join(DATA_DIR['MER2026'], 'track2_test.csv'),
}


#######################
## store global values
#######################
DEFAULT_IMAGE_PATCH_TOKEN = '<ImageHere>'
DEFAULT_AUDIO_PATCH_TOKEN = '<AudioHere>'
DEFAULT_FRAME_PATCH_TOKEN = '<FrameHere>'
DEFAULT_FACE_PATCH_TOKEN  = '<FaceHere>'
DEFAULT_MULTI_PATCH_TOKEN = '<MultiHere>'
IGNORE_INDEX = -100
