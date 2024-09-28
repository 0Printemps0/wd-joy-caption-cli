import toml
import os
from datetime import datetime
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from utils.download import download_models
from utils.image import get_image_paths, image_process, image_process_image, image_process_gbr
from utils.inference import get_caption_file_path, Llama, Joy, Tagger
from utils.inference import DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT_WITHOUT_WD, DEFAULT_USER_PROMPT_WITH_WD
from utils.logger import Logger

def load_config(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    content = content.replace('null', 'None')
    args = toml.loads(content)
    
    return args

class Caption:
    def __init__(
            self,
            args
    ):
        # Set flags

        self.use_wd = True if args['caption_method'] in ["wd+joy", "wd+llama", "wd"] else False
        self.use_joy = True if args['caption_method'] in ["wd+joy", "joy"] else False
        self.use_llama = True if args['caption_method'] in ["wd+llama", "llama"] else False

        self.wd_model_path = None
        self.wd_tags_csv_path = None

        self.image_adapter_path = None
        self.clip_path = None
        self.llm_path = None

        self.llama_path = None

        self.my_tagger = None
        self.my_joy = None
        self.my_llama = None

        # Set logger
        workspace_path = os.path.dirname(__file__)
        data_dir_path = Path(args['data_path'])

        log_file_path = data_dir_path.parent if os.path.exists(data_dir_path.parent) else workspace_path

        if args['custom_caption_save_path']:
            log_file_path = Path(args['custom_caption_save_path'])

        log_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        # caption_failed_list_file = f'Caption_failed_list_{log_time}.txt'

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
            self.my_logger = Logger(args['log_level'], log_file).logger
            self.my_logger.info(f'Set log level to "{args["log_level"]}"')
        else:
            self.my_logger = Logger('INFO', log_file).logger
            self.my_logger.warning('Invalid log level, set log level to "INFO"!')

        if args['save_logs']:
            self.my_logger.info(f'Log file will be saved as "{log_file}".')

    def download_models(
            self,
            args
    ):
        # Set models save path
        if os.path.exists(Path(args['models_save_path'])):
            models_save_path = Path(args['models_save_path'])
        else:
            models_save_path = Path(os.path.join(Path(__file__).parent, args['models_save_path']))

        if self.use_wd:
            # Check wd models path from json
            if args['wd_config'] is None:
                wd_config_file = os.path.join(Path(__file__).parent, 'configs', 'default_wd.json')
            else:
                wd_config_file = os.path.join(Path(__file__).parent, 'configs', args['wd_config'])

            # Download wd models
            self.wd_model_path, self.wd_tags_csv_path = download_models(
                logger=self.my_logger,
                models_type="wd",
                args=args,
                config_file=wd_config_file,
                models_save_path=models_save_path,
            )

        if self.use_joy:
            # Check joy models path from json
            if args['llm_config'] is None:
                llm_config_file = os.path.join(Path(__file__).parent, 'configs', 'default_joy.json')
            else:
                llm_config_file = os.path.join(Path(__file__).parent, 'configs', args['llm_config'])

            # Download joy models
            self.image_adapter_path, self.clip_path, self.llm_path = download_models(
                logger=self.my_logger,
                models_type="joy",
                args=args,
                config_file=llm_config_file,
                models_save_path=models_save_path,
            )

        elif self.use_llama:
            # Check joy models path from json
            if args['llm_config'] is None:
                llm_config_file = os.path.join(Path(__file__).parent, 'configs', 'default_llama_3.2V.json')
            else:
                llm_config_file = os.path.join(Path(__file__).parent, 'configs', args['llm_config'])

            # Download joy models
            self.llama_path = download_models(
                logger=self.my_logger,
                models_type="llama",
                args=args,
                config_file=llm_config_file,
                models_save_path=models_save_path,
            )

    def load_models(
            self,
            args
    ):
        if self.use_wd:
            # Load wd models
            self.my_tagger = Tagger(
                logger=self.my_logger,
                args=args,
                model_path=self.wd_model_path,
                tags_csv_path=self.wd_tags_csv_path
            )
            self.my_tagger.load_model()

        if self.use_joy:
            # Load joy models
            self.my_joy = Joy(
                logger=self.my_logger,
                args=args,
                image_adapter_path=self.image_adapter_path,
                clip_path=self.clip_path,
                llm_path=self.llm_path
            )
            self.my_joy.load_model()

        elif self.use_llama:
            self.my_llama = Llama(
                logger=self.my_logger,
                args=args,
                llm_path=self.llama_path,
            )
            self.my_llama.load_model()

    def run_inference(
            self,
            args
    ):
        # Inference
        # Set llm system prompt
        if args['llm_system_prompt'] == "DEFAULT_SYSTEM_PROMPT":
            args['llm_system_prompt'] = DEFAULT_SYSTEM_PROMPT
        # Set llm user prompt
        if not args['llm_user_prompt']:
                args['llm_user_prompt'] = DEFAULT_USER_PROMPT_WITHOUT_WD
        if args['llm_user_prompt'] == DEFAULT_USER_PROMPT_WITHOUT_WD:
            if not args['llm_caption_without_wd']:
                self.my_logger.info(f"LLM user prompt not defined, using default version with wd tags...")
                args['llm_user_prompt'] = DEFAULT_USER_PROMPT_WITH_WD

        if self.use_wd and (self.use_joy or self.use_llama):
            # run
            if args['run_method']=="sync":
                image_paths = get_image_paths(logger=self.my_logger,path=Path(args['data_path']),recursive=args['recursive'])
                pbar = tqdm(total=len(image_paths), smoothing=0.0)
                for image_path in image_paths:
                    try:
                        pbar.set_description('Processing: {}'.format(image_path if len(image_path) <= 40 else
                                                                     image_path[:15]) + ' ... ' + image_path[-20:])
                        # Caption file
                        wd_caption_file = get_caption_file_path(
                            self.my_logger,
                            data_path=args['data_path'],
                            image_path=Path(image_path),
                            custom_caption_save_path=args['custom_caption_save_path'],
                            caption_extension=args['wd_caption_extension']
                        )
                        llm_caption_file = get_caption_file_path(
                            self.my_logger,
                            data_path=args['data_path'],
                            image_path=Path(image_path),
                            custom_caption_save_path=args['custom_caption_save_path'],
                            caption_extension=args['llm_caption_extension']
                        )
                        # image to pillow
                        image = Image.open(image_path)
                        tag_text = ""

                        if not (args['wd_file_action'] == "skip" and os.path.isfile(wd_caption_file)):
                            # WD Caption
                            wd_image = image_process(image, self.my_tagger.model_shape_size)
                            self.my_logger.debug(f"Resized image shape: {wd_image.shape}")
                            wd_image = image_process_gbr(wd_image)
                            tag_text, rating_tag_text, character_tag_text, general_tag_text = self.my_tagger.get_tags(
                                image=wd_image
                            )

                            if args['wd_file_action'] == "overwrite":
                                # Write WD Caption
                                with open(wd_caption_file, "wt", encoding="utf-8") as f:
                                    f.write(tag_text)
                                    self.my_logger.warning(f'wd_file_action is set to overwrite!!!')
                                    self.my_logger.debug(f"Image path: {image_path}")
                                    self.my_logger.debug(f"WD Caption path: {wd_caption_file}")
                                    self.my_logger.debug(f"WD Caption content: {tag_text}")
                            elif args['wd_file_action'] == "prepend":
                                with open(wd_caption_file, "rt", encoding="utf-8") as f:
                                    existing_content = f.read()
                                with open(wd_caption_file, "wt", encoding="utf-8") as f:
                                    f.write(tag_text + existing_content)
                                    self.my_logger.warning(f'wd_file_action is set to prepend!!!')
                                    self.my_logger.debug(f"Image path: {image_path}")
                                    self.my_logger.debug(f"WD Caption path: {wd_caption_file}")
                                    self.my_logger.debug(f"WD Caption content: {tag_text}")
                            elif args['wd_file_action'] == "append":
                                with open(wd_caption_file, "at", encoding="utf-8") as f:
                                    f.write(tag_text)
                                    self.my_logger.warning(f'wd_file_action is set to append!!!')
                                    self.my_logger.debug(f"Image path: {image_path}")
                                    self.my_logger.debug(f"WD Caption path: {wd_caption_file}")
                                    self.my_logger.debug(f"WD Caption content: {tag_text}")
                            elif args['wd_file_action'] == "skip" and not os.path.isfile(wd_caption_file):
                                with open(wd_caption_file, "wt", encoding="utf-8") as f:
                                    f.write(tag_text)
                                    self.my_logger.debug(f"Image path: {image_path}")
                                    self.my_logger.debug(f"WD Caption path: {wd_caption_file}")
                                    self.my_logger.debug(f"WD Caption content: {tag_text}")
                            if args['wd_model_name'].lower().startswith("wd"):
                                self.my_logger.debug(f"WD Rating tags: {rating_tag_text}")
                                self.my_logger.debug(f"WD Character tags: {character_tag_text}")
                            self.my_logger.debug(f"WD General tags: {general_tag_text}")
                        else:
                            self.my_logger.warning(f'wd_file_action is set to skip!!! '
                                                   f'WD Caption file {wd_caption_file} already exists, '
                                                   f'Skip this caption.')

                        if not (args['llm_file_action'] == "skip" and os.path.isfile(llm_caption_file)):
                            # LLM
                            llm_image = image_process(image, args['image_size'])
                            self.my_logger.debug(f"Resized image shape: {llm_image.shape}")
                            llm_image = image_process_image(llm_image)
                            # LLM Caption
                            caption = ""
                            if self.use_joy:
                                caption = self.my_joy.get_caption(
                                    image=llm_image,
                                    user_prompt=str(f'{args["llm_user_prompt"]}{tag_text}\n') if not args['llm_caption_without_wd'] else str(f'{args["llm_user_prompt"]}\n'),
                                    temperature=args['llm_temperature'],
                                    max_new_tokens=args['llm_max_tokens']
                                )
                            elif self.use_llama:
                                caption = self.my_llama.get_caption(
                                    image=llm_image,
                                    system_prompt=DEFAULT_SYSTEM_PROMPT if args['llm_system_prompt'] == DEFAULT_SYSTEM_PROMPT else args['llm_system_prompt'],
                                    user_prompt=str(f'{args["llm_user_prompt"]}{tag_text}\n') if not args['llm_caption_without_wd'] else str(f'{args["llm_user_prompt"]}\n'),
                                    temperature=args['llm_temperature'],
                                    max_new_tokens=args['llm_max_tokens']
                                )
                            caption = caption.replace('\n', '')

                            if args['llm_file_action'] == "overwrite":
                                # Write LLM Caption
                                with open(llm_caption_file, "wt", encoding="utf-8") as f:
                                    f.write(caption)
                                    self.my_logger.warning(f'llm_file_action is set to overwrite!!!')
                                    self.my_logger.debug(f"Image path: {image_path}")
                                    self.my_logger.debug(f"LLM Caption path: {llm_caption_file}")
                                    self.my_logger.debug(f"LLM Caption content: {caption}")
                            elif args['llm_file_action'] == "prepend":
                                with open(llm_caption_file, "rt", encoding="utf-8") as f:
                                    existing_content = f.read()
                                with open(llm_caption_file, "wt", encoding="utf-8") as f:
                                    f.write(caption + existing_content)
                                    self.my_logger.warning(f'llm_file_action is set to prepend!!!')
                                    self.my_logger.debug(f"Image path: {image_path}")
                                    self.my_logger.debug(f"LLM Caption path: {llm_caption_file}")
                                    self.my_logger.debug(f"LLM Caption content: {caption}")
                            elif args['llm_file_action'] == "append":
                                with open(llm_caption_file, "at", encoding="utf-8") as f:
                                    f.write(caption)
                                    self.my_logger.warning(f'llm_file_action is set to append!!!')
                                    self.my_logger.debug(f"Image path: {image_path}")
                                    self.my_logger.debug(f"LLM Caption path: {llm_caption_file}")
                                    self.my_logger.debug(f"LLM Caption content: {caption}")
                            elif args['llm_file_action'] == "skip" and not os.path.isfile(llm_caption_file):
                                with open(llm_caption_file, "wt", encoding="utf-8") as f:
                                    f.write(caption)
                                    self.my_logger.debug(f"Image path: {image_path}")
                                    self.my_logger.debug(f"LLM Caption path: {llm_caption_file}")
                                    self.my_logger.debug(f"LLM Caption content: {caption}")
                        else:
                            self.my_logger.warning(f'llm_file_action is set to skip!!! '
                                                   f'LLM Caption file {llm_caption_file} already exists, '
                                                   f'Skip this caption.')

                    except Exception as e:
                        self.my_logger.error(f"Failed to caption image: {image_path}, skip it.\nerror info: {e}")
                        continue

                    pbar.update(1)

                pbar.close()

                if args['wd_tags_frequency']:
                    sorted_tags = sorted(self.my_tagger.tag_freq.items(), key=lambda x: x[1], reverse=True)
                    self.my_logger.info('WD Tag frequencies:')
                    for tag, freq in sorted_tags:
                        self.my_logger.info(f'{tag}: {freq}')
            else:
                pbar = tqdm(total=2, smoothing=0.0)
                pbar.set_description('Processing with WD model...')
                self.my_tagger.inference()
                pbar.update(1)
                if self.use_joy:
                    pbar.set_description('Processing with joy model...')
                    self.my_joy.inference()
                    pbar.update(1)
                elif self.use_llama:
                    pbar.set_description('Processing with Llama model...')
                    self.my_llama.inference()
                    pbar.update(1)
                pbar.close()
        else:
            if self.use_wd:
                self.my_tagger.inference()
            if self.use_joy and not self.use_llama:
                self.my_joy.inference()
            elif not self.use_joy and self.use_llama:
                self.my_llama.inference()

    def unload_models(
            self
    ):
        # Unload models
        if self.use_wd:
            self.my_tagger.unload_model()
        if self.use_joy:
            self.my_joy.unload_model()
        if self.use_llama:
            self.my_llama.unload_model()

if __name__ == "__main__":
    args = load_config('config.toml')
    my_caption = Caption(args)
    my_caption.download_models(args)
    my_caption.load_models(args)
    my_caption.run_inference(args)
    my_caption.unload_models()