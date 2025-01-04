import groq
from groq import Groq

from core.constants import EVALUATION_BACKUP_PATH
from core.utils.logger import logger
from core.utils.file_utils import load_evaluation_backup, save_evaluation_backup
from core.utils.texts_processing import *


class TextsEstimator:
    def __init__(self, api_key: str, prompt_file_link: str, models: list, output_filename: str):
        self.client = Groq(
            api_key=api_key
        )
        self.models = models
        self.output_path = f"output/{output_filename}"
        with open(prompt_file_link, 'r', encoding='utf-8') as f:
            self.prompt = f.read()

    def evaluate_texts(self, data_path: str, max_size: int):
        input_df = create_input_dataframe(data_path=data_path)
        evaluation_backup = load_evaluation_backup(EVALUATION_BACKUP_PATH)

        input_df['already_evaluated'] = input_df['id'].apply(lambda file_number: str(file_number) in evaluation_backup)
        to_evaluate = input_df[~input_df['already_evaluated']].copy()
        already_evaluated = input_df[input_df['already_evaluated']].copy()

        for index, row in already_evaluated.iterrows():
            file_id = str(row['id'])
            scores = evaluation_backup[file_id]['scores'] if file_id in evaluation_backup else {}
            for model in self.models:
                already_evaluated.at[index, model] = scores.get(model, None)

        if max_size:
            large_files_df = to_evaluate[to_evaluate['rounded_size_kb'] > max_size]
            to_evaluate = to_evaluate[to_evaluate['rounded_size_kb'] <= max_size]
            logger.info(f"Файлов, больше {max_size} Кб - {len(large_files_df)}, "
                        f"уже оцененных файлов: {len(already_evaluated)}, "
                        f"на оценку будет отправлено файлов: {len(to_evaluate)}")
            for model in self.models:
                large_files_df[model] = 5
        for model in self.models:
            to_evaluate[model] = None

        for index, row in to_evaluate.iterrows():
            full_prompt = self.prompt.replace("###", row['content']).replace("***", row['filename'])
            evaluation_summary = self.evaluate_page(full_prompt)
            for model, score in evaluation_summary.items():
                to_evaluate.at[index, model] = score
            evaluation_backup[str(row['id'])] = {
                'filename': row['filename'],
                'scores': evaluation_summary
            }
            save_evaluation_backup(evaluation_backup, EVALUATION_BACKUP_PATH)

        if max_size:
            already_evaluated = pd.concat([already_evaluated, large_files_df], ignore_index=True)

        full_df = pd.concat([to_evaluate, already_evaluated], ignore_index=True) if len(to_evaluate) > 0 \
            else already_evaluated
        model_columns = [model for model in self.models if model in full_df.columns]
        full_df['average_score'] = full_df[model_columns].mean(axis=1, skipna=True)
        full_df.to_csv(self.output_path)
        logger.info("Оценка файлов завершена успешно")

    def evaluate_page(self, full_prompt: str) -> dict:
        evaluation_summary = {}
        for model in self.models:
            try:
                completion = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": "Ты - ассистент-разметчик данных для дообучения LLM интеллектуального ассистента Донецкого Государственного университета"
                        },
                        {
                            "role": "user",
                            "content": full_prompt
                        }
                    ],
                    temperature=1,
                    max_tokens=1024,
                    top_p=1,
                    stop=None,
                    stream=True,
                    seed=42
                )
                for chunk in completion:
                    content = chunk.choices[0].delta.content
                    if reply_is_valid(content):
                        try:
                            mark = int(content)
                            evaluation_summary[model] = mark
                        except ValueError:
                            logger.error(f"Ошибка, ответ модели {model} - {content} "
                                         f"не может быть интерпретирован как число")
            except groq.NotFoundError as e:
                logger.error(f"{e}")
        logger.info(evaluation_summary)
        return evaluation_summary
