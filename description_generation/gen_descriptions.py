import os
import json
import yaml
import argparse

from openai import OpenAI

from utils import imagnet_prompts
from utils import imagenet_variants
from utils import cls_names_video
from utils import cls_to_names

client = OpenAI(
        base_url="https://api.openai.com/v1",
        api_key="YOUR_API_KEY",
    )


def get_imagenetX_classnames(set_id):
    assert set_id in ['imagenet_a', 'imagenet_r', 'imagenet_sketch', 'imagenetv2', 'imagenet']
    classnames_all = imagnet_prompts.imagenet_classes
    classnames = []
    if set_id in ['imagenet_a', 'imagenet_r', 'imagenetv2']:
        if set_id == 'imagenetv2':
            set_id = 'imagenet_v'
        label_mask = eval("imagenet_variants.{}_mask".format(set_id))
        if set_id == 'imagenet_r':
            for i, m in enumerate(label_mask):
                if m:
                    classnames.append(classnames_all[i])
        else:
            classnames = [classnames_all[i] for i in label_mask]
    else:
        classnames = classnames_all
    return classnames


def get_classnames(dataset_name):
    if dataset_name in ['imagenet_a', 'imagenet_r', 'imagenet_sketch', 'imagenetv2', 'imagenet']:
        return get_imagenetX_classnames(dataset_name)
    elif dataset_name in ['hmbd51', 'ucf101v', 'k600']:
        return eval(f"cls_names_video.{dataset_name.lower()}_classes")
    else:
        return eval(f"cls_to_names.{dataset_name.lower()}_classes")


def split_answer(s):
    lines = s.strip().split('\n')
    result = []
    for line in lines:
        if line.startswith('- '):
            line = line[2:]
        sub_items = line.split('. ')
        result.extend(sub_items)
    
    if result and not result[-1].endswith('.'):
        result[-1] += '.'
    
    result = [item + '. ' if item and not item.endswith('.') else item for item in result]
    return result


def get_cls_descs_single_question(args, classname, question):
    assert args.llm in ["gpt-3.5-turbo", "gpt-4o", "qwen-plus"]
    
    SYS_PROMPT ='You are a helpful Q&A assistant to describe or identify something. '                                       \
                'Answer in several INDEPENDENT sentences like the following examples. '                                     \
                'Respond without using any pronouns. '                                                                      \
                'Instead, try to refer to individuals or objects using their names or by using other descriptive terms. '   \
                'Also, provide answers in the form of independent sentences. '                                              \
                'Each sentence should be capable of standing alone as a description of something specific. '                \
                'Each line of your answer should be in one sentence and not too long.'
    
    EXAMPLE_Q_1 = "Q: How do you identify a Lemur?"
    EXAMPLE_A_1 =   "- Lemurs are small to medium-sized primates.\n"                                                        \
                    "- Lemurs typically have a pointed snout and large, round eyes.\n"                                      \
                    "- Lemurs often have a distinctive fur pattern, with variations of gray, brown, black, and white.\n"    \
                    "- Lemurs have long tails, which they use for balance and communication.\n"                             \
                    "- Many lemurs have a specialized grooming claw on their second toe.\n"                                 \
                    "- Lemurs are arboreal creatures, meaning they spend most of their time in trees."
    
    EXAMPLE_Q_2 = "Q: Describe what a Television looks like."
    EXAMPLE_A_2 =   "- A television typically has a flat, rectangular screen.\n"                                                        \
                    "- A television often has a black or dark-colored frame surrounding the screen.\n"                                  \
                    "- On the front of a television, there are control buttons or a touch-sensitive panel for adjusting settings.\n"    \
                    "- Behind the screen of a television, there are various electronic components.\n"                                   \
                    "- A television usually has speakers located either on the sides or at the bottom.\n"                               \
                    "- A television is often mounted on a stand or attached to a wall."
    
    response = client.chat.completions.create(
        model=args.llm,
        messages=[
            {"role":"system", "content":SYS_PROMPT},
            
            {"role":"user", "content":EXAMPLE_Q_1},
            {"role":"assistant", "content":EXAMPLE_A_1},
            
            {"role":"user", "content":EXAMPLE_Q_2},
            {"role":"assistant", "content":EXAMPLE_A_2},
            
            {"role":"user", "content":f"{question.format(classname)}"},
        ],
    )
    answer = response.choices[0].message.content
    descs = split_answer(answer)
    return descs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm', type=str, default='gpt-3.5-turbo', help='The LLM-API to use for generation.')
    parser.add_argument('--prompt_folder', type=str, default='Prompts', help='The root folder to save all generated prompts.')
    parser.add_argument('--question_file', type=str, default='0_Dataset_Questions.yaml', help='The file name of generated Step-1 dataset questions.')
    parser.add_argument('--desc_count', type=int, default=50, help='Minimum number of descriptions for each class to generate.')
    args = parser.parse_args()
    
    save_dir = os.path.join(args.prompt_folder, args.llm)
    question_path = os.path.join(save_dir, args.question_file)
    with open(question_path, 'r') as f:
        dataset_questions = yaml.load(f, yaml.FullLoader)
    
    for dataset_name, questions in dataset_questions.items():
        print(f"{dataset_name}: {len(get_classnames(dataset_name))} classes to generate.")
        
        data = {}
        for classname in get_classnames(dataset_name):
            # WARNING: if `len(results)` is under `args.desc_count`, it will (always) keep querying
            #          watch output log or set a maximum retry times if you want
            descriptions = []
            gen_questions = iter(questions)
            while len(descriptions) < args.desc_count:
                try:
                    question = next(gen_questions)
                    single_descs = get_cls_descs_single_question(args, classname, question)
                    descriptions.extend(single_descs)
                except StopIteration:
                    gen_questions = iter(questions)
                print(f"    Generating for {dataset_name}'s {classname}. {len(descriptions)} descriptions generated.")
            data.update({classname: descriptions})
        
        save_path = os.path.join(save_dir, f'{dataset_name}.json')
        with open(save_path, 'w') as f:
            json.dump(data, f)


