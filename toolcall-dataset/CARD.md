---
license: apache-2.0
language:
- tr
task_categories:
- text-generation
- question-answering
tags:
- function-calling
- tool-use
- mathematics
- turkish
- synthetic
- reasoning
- sharegpt
- unsloth
size_categories:
- 1K<n<10K
dataset_info:
  features:
  - name: conversations
    list:
    - name: from
      dtype: string
    - name: value
      dtype: string
  splits:
  - name: train
    num_bytes: 1669009
    num_examples: 1207
  download_size: 1668167
  dataset_size: 1669009
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---

# math-toolcall-tr

Türkçe **matematik odaklı fonksiyon çağırma (tool calling)** veri seti — 1.207 örnek,
ShareGPT formatında, düşünme adımları (`<think>`) dahil.

> **EN:** A Turkish synthetic dataset for training LLMs to call math functions correctly
> and present the results clearly. 1,207 ShareGPT-format conversations with reasoning
> traces, covering 70 math topics and 8 tool-calling scenarios.

## Ne öğretir

İki beceriyi birlikte hedefler:

1. **Doğru araç seçimi ve parametre çıkarımı** — gereksiz çağrı yapmamak dahil.
2. **Ham sayısal çıktıyı kullanıcıya doğal dille anlatmak.**

Veri seti kasıtlı olarak sadece "başarılı çağrı" öğretmez. Örneklerin **%31'inde hiç
araç çağrılmaz**: ya soru zaten bilgiyle cevaplanabilir, ya da zorunlu bir parametre
eksiktir ve model uydurmak yerine netleştirme sorusu sorar.

## Format

Her satır bir konuşma. Araç çağıran örnekler **4 turlu**:

| Tur | Rol | İçerik |
|---|---|---|
| 1 | `human` | kullanıcı sorusu |
| 2 | `gpt` | `<think>…</think>` + `<tool_call>{"name":…,"arguments":…}</tool_call>` |
| 3 | `human` | `<tool_response>{…}</tool_response>` |
| 4 | `gpt` | doğal dille son cevap |

Araç gerektirmeyen örneklerde tek `gpt` turu olur: `<think>` + cevap.

Bu kurgu, `train_on_responses_only` ile eğitildiğinde modelin **araç sonucunu uydurmayı
değil, yorumlamayı** öğrenmesini sağlar — kayıp yalnızca `gpt` turlarında hesaplanır,
`tool_response` turu maskelidir.

### Örnek

```json
[
  {"from": "human",
   "value": "Atölye için ayırdığım 12 bin lira iki yıl %14 büyürse, enflasyon %11 olursa gerçek karşılığı ne olur?"},
  {"from": "gpt",
   "value": "<think>\nKullanıcı hem nominal birikimi hem enflasyona göre reel alım gücünü soruyor…\n</think>\n\n<tool_call>{\"name\": \"compound_interest\", \"arguments\": {\"principal\": \"12000\", \"annual_rate\": \"0.14\", \"years\": \"2\", \"inflation_rate\": \"0.11\"}}</tool_call>"},
  {"from": "human",
   "value": "<tool_response>{\"nominal_amount\": \"15595.20\", \"real_amount\": \"12657.41\"}</tool_response>"},
  {"from": "gpt",
   "value": "12 bin lira, yılda %14 getiriyle 2 yıl sonunda nominal olarak 15.595,20 TL olur…"}
]
```

## Kapsam

**13 alt alan, 70 konunun tamamı** temsil edilir.

| Alt alan | Örnek | Alt alan | Örnek |
|---|---|---|---|
| cebir | 154 | olasilik_kombinatorik | 79 |
| analiz | 116 | birim_olcu | 75 |
| finansal_matematik | 112 | istatistik | 75 |
| geometri | 110 | ayrik_matematik | 60 |
| aritmetik | 106 | optimizasyon | 50 |
| trigonometri | 100 | | |
| sayi_teorisi | 90 | | |
| lineer_cebir | 80 | | |

### Senaryolar

| Senaryo | Örnek | Ne öğretir |
|---|---|---|
| `arac_gereksiz` | 186 | Araç gerekmiyorsa çağırmamak |
| `tek_cagri` | 170 | Tek aracı doğru parametreyle çağırmak |
| `hata_yonetimi` | 161 | Sıfıra bölme / tanımsızlık gibi hataları açıklamak |
| `yanlis_arac_tuzagi` | 160 | Benzer isimli araçlar arasından doğrusunu seçmek |
| `eksik_parametre` | 151 | Uydurmak yerine netleştirme sorusu sormak |
| `cok_adimli_gorev` | 134 | Tek istekte 3+ çağrıyla tamamlamak |
| `paralel_cagri` | 130 | Bağımsız hesapları aynı anda yapmak |
| `zincirli_cagri` | 115 | İkinci çağrının girdisini ilkinin sonucundan almak |

**Zorluk:** kolay 456 · orta 346 · zor 405
**Çoklu çağrı içeren örnek:** 406

## Kullanım

```python
from datasets import load_dataset
dataset = load_dataset("bilalabic/math-toolcall-tr", split="train")
```

Unsloth ile (ShareGPT formatı doğrudan desteklenir):

```python
from unsloth.chat_templates import standardize_data_formats
dataset = standardize_data_formats(dataset)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    return {"text": [tokenizer.apply_chat_template(c, tokenize=False).removeprefix("<bos>")
                     for c in convos]}

dataset = dataset.map(formatting_prompts_func, batched=True)
```

Eğitimde `train_on_responses_only` kullanılması önerilir — kullanıcı ve `tool_response`
turları maskelenir, kayıp yalnızca model yanıtlarında hesaplanır.

## Üretim yöntemi

İki aşamalı sentetik üretim, tamamı **`gpt-5.4-mini`** ile (reasoning effort: medium):

1. **Soru üretimi** — her istek için (alt alan, konu, senaryo, zorluk) kombinasyonu
   seçilir; model 2-4 fonksiyon şeması + doğal bir kullanıcı mesajı üretir.
   Şemaların yalnızca 1-2'si gerçekten gereklidir, diğerleri inandırıcı çeldiricidir.
2. **Cevap üretimi** — aynı model düşünme adımlarını, araç çağrılarını, beklenen araç
   sonuçlarını ve son yanıtı üretir.

Tekrar eden sorular elenir. Konu dağılımı, işlenmemiş konular önceliklendirilerek
70/70 kapsamaya ulaştırılmıştır.

Üretim kodunun tamamı açık: [github.com/BilalAbic/math-toolcall-tr](https://github.com/BilalAbic/math-toolcall-tr)

## İlgili model

| | |
|---|---|
| 🤖 LoRA adaptörü | [bilalabic/gemma_4_math-toolcall-tr_lora](https://huggingface.co/bilalabic/gemma_4_math-toolcall-tr_lora) |
| 📓 Eğitim notebook'u | [gemma4_e4b_math_toolcall_lora.ipynb](https://github.com/BilalAbic/math-toolcall-tr/blob/main/notebooks/gemma4_e4b_math_toolcall_lora.ipynb) |
| 💻 GitHub | https://github.com/BilalAbic/math-toolcall-tr |

Gemma-4 E4B üzerine LoRA (r=8) ile eğitilmiştir; 3 epoch, `train_on_responses_only`.

> ⚠️ **Sürüm farkı:** Bu adaptör, veri setinin **757 örneklik daha eski bir anlık
> görüntüsüyle** eğitilmiştir — buradaki güncel 1.207 örnekle **değil**. Güncel sürümle
> yeniden eğitim henüz yapılmamıştır.

## Sınırlamalar

Bu veri seti sentetiktir ve **insan doğrulamasından geçmemiştir**. Kullanmadan önce
bilinmesi gerekenler:

- **Araç sonuçları gerçek değildir.** `tool_response` içerikleri gerçek bir fonksiyon
  çalıştırılarak değil, model tarafından üretilmiştir. Gerçekçi olacak şekilde
  yönlendirilmiştir ancak doğrulukları garanti edilemez.
- **Matematiksel hatalar mümkündür.** Prompt matematiksel doğruluğu şart koşar ve
  incelenen örneklerde hesaplar doğrudur, ancak tüm örnekler tek tek doğrulanmamıştır.
- **Senaryo tutarsızlığı: %3,1.** 37 örnekte araç çağrısı beklenirken çağrı listesi
  boştur (çoğunlukla `hata_yonetimi` senaryosunda). Kritik uygulamalarda bu örnekler
  filtrelenebilir.
- **Tek model kaynaklı.** Tamamı tek bir modelden üretildiği için o modelin üslup ve
  akıl yürütme kalıplarını taşır.
- **Sadece Türkçe ve sadece matematik.** Başka alanlara veya dillere genellemez.

## Alıntı

```bibtex
@misc{math-toolcall-tr,
  title  = {math-toolcall-tr: Turkish Math Tool-Calling Dataset},
  author = {Bilal Abiç},
  year   = {2026},
  url    = {https://huggingface.co/datasets/bilalabic/math-toolcall-tr}
}
```
