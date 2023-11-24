# Cesar: Um modelo de recomendação conversacional em português brasileiro

_This paper introduces the Cesar model, a novel language model for conversational recommendation. In a landscape
where personalized recommendations play a crucial role in widely-used platforms, such as social media and streaming
services, addressing the challenge of efficiently capturing user-specific desires remains paramount. Cesar tackles this gap
by employing advanced natural language processing and transfer learning techniques, adapting to the context of movie
recommendations in Brazilian Portuguese. Unlike traditional approaches relying on implicit user features like likes and watch
time, Cesar can understand and respond to more specific queries, such as preferences for genre, duration, and specific cast
members. Furthermore, the model addresses the "Cold Start" challenge by leveraging prior knowledge to provide meaningful
recommendations from the user’s initial interactions. Developed within the accessible Google Colab environment, Cesar’s design
facilitates future adaptations for high availability environments. Results indicate that Cesar presents a promising approach
to enhance user experience in conversational recommendation systems, offering improved refinement, dynamic interaction,
and explainability in generating recommendation lists. This work significantly contributes to expanding the conversational
recommendation theme to new languages, advancing the understanding and application of language models in specific
contexts, thereby enriching user-platform interactions._

# Carregar modelo
```python
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM
config = PeftConfig.from_pretrained("matheusrdgsf/cesar-ptbr")
model = AutoModelForCausalLM.from_pretrained("TheBloke/zephyr-7B-beta-GPTQ", revision="gptq-8bit-32g-actorder_True", device_map='auto')
model = PeftModel.from_pretrained(model, "matheusrdgsf/cesar-ptbr")
```

# Inferência
```python
from transformers import GenerationConfig
from transformers import AutoTokenizer
tokenizer_model = AutoTokenizer.from_pretrained('TheBloke/zephyr-7B-beta-GPTQ')
tokenizer_template = AutoTokenizer.from_pretrained('HuggingFaceH4/zephyr-7b-alpha')
generation_config = GenerationConfig(
    do_sample=True,
    temperature=0.1,
    top_p=0.25,
    top_k=0,
    max_new_tokens=512,
    repetition_penalty=1.1,
    eos_token_id=tokenizer_model.eos_token_id,
    pad_token_id=tokenizer_model.eos_token_id,
)
def get_inference(
    text,
    model,
    tokenizer_model=tokenizer_model,
    tokenizer_template=tokenizer_template,
    generation_config=generation_config,
):
    st_time = time.time()
    inputs = tokenizer_model(
        tokenizer_template.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": "Você é um chatbot para indicação de filmes. Responda em português e de maneira educada sugestões de filmes para os usuários.",
                },
                {"role": "user", "content": text},
            ],
            tokenize=False,
        ),
        return_tensors="pt",
    ).to("cuda")
    outputs = model.generate(**inputs, generation_config=generation_config)
    print('inference time:', time.time() - st_time)
    return tokenizer_model.decode(outputs[0], skip_special_tokens=True).split('\n')[-1]
get_inference('Poderia indicar filmes de ação de até 2 horas?', model)
```

O modelo de [cesar-ptbr](https://huggingface.co/matheusrdgsf/cesar-ptbr) está disponível para utilização na Huggingface.

O dataset de treinamento [re_dial_ptbr](https://huggingface.co/datasets/matheusrdgsf/re_dial_ptbr) está disponível para utilização na Huggingface.