import toml
import os
from datetime import datetime
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from utils.download import download_joy, download_wd
from utils.image import get_image_paths, image_process, image_process_image, image_process_gbr
from utils.inference import Joy, Tagger, get_caption_file_path
from utils.logger import Logger

DEFAULT_USER_PROMPT_WITH_WD = """
As an AI experiment assistant, you need to provide descriptive text for the image data. The more accurate the description, the more helpful it is for the experiment. Please concise your description to limit the number of words and only choose the most accurate expression. Before generating the description, I will give you some tips about the following picture:
{wd_tags}.\n
"""

DEFAULT_USER_PROMPT_WITHOUT_WD = """
Write a descriptive caption for this image:\n
"""

def load_config(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    content = content.replace('null', 'None')
    config = toml.loads(content)
    
    return config

def main(args):
    # Set flags
    use_wd = True if args['caption_method'] in ["both", "wd"] else False
    use_joy = True if args['caption_method'] in ["both", "joy"] else False

    # Set logger
    workspace_path = os.getcwd()
    data_dir_path = Path(args['data_path'])

    log_file_path = data_dir_path.parent if os.path.exists(data_dir_path.parent) else workspace_path

    if args['custom_caption_save_path']:
        log_file_path = Path(args['custom_caption_save_path'])

    log_time = datetime.now().strftime('%Y%m%d_%H%M%S')

    if os.path.exists(data_dir_path):
        log_name = os.path.basename(data_dir_path)
    else:
        print(f'{data_dir_path} NOT FOUND!!!')
        raise FileNotFoundError

    if args['save_logs']:
        log_file = f'Caption_{log_name}_{log_time}.log' if log_name else f'test_{log_time}.log'
        log_file = os.path.join(log_file_path, log_file) \
            if os.path.exists(log_file_path) else os.path.join(os.getcwd(), log_file)
    else:
        log_file = None

    if str(args['log_level']).lower() in 'debug, info, warning, error, critical':
        my_logger = Logger(args['log_level'], log_file).logger
        my_logger.info(f'Set log level to "{args["log_level"]}"')
    else:
        my_logger = Logger('INFO', log_file).logger
        my_logger.warning('Invalid log level, set log level to "INFO"!')

    if args['save_logs']:
        my_logger.info(f'Log file will be saved as "{log_file}".')

    # Set models save path
    if os.path.exists(Path(args['models_save_path'])):
        models_save_path = Path(args['models_save_path'])
    else:
        models_save_path = Path(os.path.join(Path(__file__).parent, args['models_save_path']))

    if use_wd:
        # Check wd models path from json
        wd_config_file = os.path.join(Path(__file__).parent, 'configs', 'default_wd.json') \
            if args['wd_config'] == "default_wd.json" else Path(args['wd_config'])

        # Download wd models
        model_path, tags_csv_path = download_wd(
            logger=my_logger,
            config_file=wd_config_file,
            model_name=str(args['wd_model_name']),
            model_site=str(args['model_site']),
            models_save_path=models_save_path,
            use_sdk_cache=args['use_sdk_cache'],
            download_method=str(args['download_method']),
            skip_local_file_exist=args['skip_download'],
            force_download=args['force_download']
        )
    if use_joy:
        # Check joy models path from json
        joy_config_file = os.path.join(Path(__file__).parent, 'configs', 'default_joy.json') \
            if args['joy_config'] == "default_joy.json" else Path(args['joy_config'])

        # Download joy models
        image_adapter_path, clip_path, llm_path = download_joy(
            logger=my_logger,
            config_file=joy_config_file,
            model_name=str(args['joy_model_name']),
            model_site=str(args['model_site']),
            models_save_path=models_save_path,
            use_sdk_cache=args['use_sdk_cache'],
            download_method=str(args['download_method']),
            skip_local_file_exist = args['skip_download'],
            force_download = args['force_download']
        )

    if use_wd:
        # Load wd models
        my_tagger = Tagger(
            logger=my_logger,
            args=args,
            model_path=model_path,
            tags_csv_path=tags_csv_path
        )
        my_tagger.load_model()

    if use_joy:
        # Load joy models
        my_joy = Joy(
            logger=my_logger,
            args=args,
            image_adapter_path=image_adapter_path,
            clip_path=clip_path,
            llm_path=llm_path
        )
        my_joy.load_model()

    # Inference
    if use_wd and use_joy:
        # Set joy user prompt
        if args['joy_user_prompt'] == "DEFAULT_USER_PROMPT_WITHOUT_WD":
            args['joy_user_prompt'] = DEFAULT_USER_PROMPT_WITHOUT_WD

        if args['joy_user_prompt'] == DEFAULT_USER_PROMPT_WITHOUT_WD:
            if not args['joy_caption_without_wd']:
                my_logger.info(f"Joy user prompt not defined, using default version with wd tags...")
                args['joy_user_prompt'] = DEFAULT_USER_PROMPT_WITH_WD
        # run
        if args['run_method']=="sync":
            image_paths = get_image_paths(logger=my_logger,path=Path(args['data_path']),recursive=args['recursive'])
            pbar = tqdm(total=len(image_paths), smoothing=0.0)
            for image_path in image_paths:
                try:
                    pbar.set_description('Processing: {}'.format(image_path if len(image_path) <= 40 else
                                                                 image_path[:15]) + ' ... ' + image_path[-20:])
                    image = Image.open(image_path)
                    # WD
                    wd_image = image_process(image, my_tagger.model_shape_size)
                    my_logger.debug(f"Resized image shape: {wd_image.shape}")
                    wd_image = image_process_gbr(wd_image)
                    tag_text, rating_tag_text, character_tag_text, general_tag_text = my_tagger.get_tags(
                        image=wd_image
                    )
                    wd_config_file = get_caption_file_path(
                        my_logger,
                        data_path=args['data_path'],
                        image_path=Path(image_path),
                        custom_caption_save_path=args['custom_caption_save_path'],
                        caption_extension=args['wd_caption_extension']
                    )
                    if args['not_overwrite'] and os.path.isfile(wd_config_file):
                        my_logger.warning(f'WD Caption file {wd_config_file} already exist! Skip this caption.')
                        continue

                    with open(wd_config_file, "wt", encoding="utf-8") as f:
                        f.write(tag_text + "\n")

                    my_logger.debug(f"Image path: {image_path}")
                    my_logger.debug(f"WD Caption path: {wd_config_file}")
                    if args['wd_model_name'].lower().startswith("wd"):
                        my_logger.debug(f"WD Rating tags: {rating_tag_text}")
                        my_logger.debug(f"WD Character tags: {character_tag_text}")
                    my_logger.debug(f"WD General tags: {general_tag_text}")

                    # Joy
                    joy_image = image_process(image, args['image_size'])
                    my_logger.debug(f"Resized image shape: {joy_image.shape}")
                    joy_image = image_process_image(joy_image)
                    caption = my_joy.get_caption(
                        image=joy_image,
                        user_prompt=str(args['joy_user_prompt']).format(wd_tags=tag_text),
                        temperature=args['joy_temperature'],
                        max_new_tokens=args['joy_max_tokens']
                    )
                    joy_caption_file = get_caption_file_path(
                        my_logger,
                        data_path=args['data_path'],
                        image_path=Path(image_path),
                        custom_caption_save_path=args['custom_caption_save_path'],
                        caption_extension=args['joy_caption_extension']
                    )
                    if args['not_overwrite'] and os.path.isfile(joy_caption_file):
                        my_logger.warning(f'Caption file {joy_caption_file} already exist! Skip this caption.')
                        continue

                    with open(joy_caption_file, "wt", encoding="utf-8") as f:
                        f.write(caption + "\n")
                        my_logger.debug(f"Image path: {image_path}")
                        my_logger.debug(f"Joy Caption path: {joy_caption_file}")
                        my_logger.debug(f"Joy Caption content: {caption}")

                except Exception as e:
                    my_logger.error(f"Failed to caption image: {image_path}, skip it.\nerror info: {e}")
                    continue

                pbar.update(1)

            pbar.close()

            if args['wd_tags_frequency']:
                sorted_tags = sorted(my_tagger.tag_freq.items(), key=lambda x: x[1], reverse=True)
                my_logger.info('WD Tag frequencies:')
                for tag, freq in sorted_tags:
                    my_logger.info(f'{tag}: {freq}')
        else:
            pbar = tqdm(total=2, smoothing=0.0)
            pbar.set_description('Processing with WD model...')
            my_tagger.inference()
            pbar.update(1)
            pbar.set_description('Processing with WD model...')
            my_joy.inference()
            pbar.update(1)
            pbar.close()
    else:
        if use_wd and not use_joy:
            my_tagger.inference()
        elif not use_wd and use_joy:
            my_joy.inference()

    if use_wd:
        # Unload models
        my_tagger.unload_model()
    if use_joy:
        # Unload models
        my_joy.unload_model()

if __name__ == "__main__":
    config = load_config('config.toml')
    main(config)