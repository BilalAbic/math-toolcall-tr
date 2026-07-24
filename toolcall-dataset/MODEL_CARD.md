---
base_model: unsloth/gemma-4-e4b-it-unsloth-bnb-4bit
library_name: peft
license: apache-2.0
language:
- tr
datasets:
- bilalabic/math-toolcall-tr
pipeline_tag: text-generation
tags:
- unsloth
- gemma4
- trl
- peft
- lora
- function-calling
- tool-use
- mathematics
- turkish
- text-generation-inference
- transformers
---

# gemma_4_math-toolcall-tr_lora

Türkçe **matematik fonksiyon çağırma (tool calling)** için Gemma-4 E4B üzerine eğitilmiş
LoRA adaptörü.

> **EN:** LoRA adapter fine-tuned on Gemma-4 E4B for Turkish math tool-calling — deciding
> which math function to call, extracting its arguments, and explaining the result in
> natural language.

- **Geliştiren:** bilalabic
- **Temel model:** [`unsloth/gemma-4-e4b-it-unsloth-bnb-4bit`](https://huggingface.co/unsloth/gemma-4-e4b-it-unsloth-bnb-4bit)
- **Veri seti:** [`bilalabic/math-toolcall-tr`](https://huggingface.co/datasets/bilalabic/math-toolcall-tr)
- **Depo içeriği:** LoRA adaptörü (`adapter_model.safetensors`) + tokenizer. Birleştirilmiş
  (merged) ağırlık içermez; temel model ayrıca indirilir.

## Ne yapar

Modele matematik fonksiyonları sunulduğunda:

1. **Doğru aracı seçer ve parametreleri kullanıcı mesajından çıkarır.**
2. **Gerekmiyorsa araç çağırmaz** — soru doğrudan cevaplanabiliyorsa açıklar, zorunlu bir
   parametre eksikse uydurmak yerine netleştirme sorusu sorar.
3. **Araç sonucunu doğal Türkçeyle anlatır** — ham JSON yapıştırmaz, sayıları
   birimleriyle cümle içinde verir.

Ayrıca yanıtlarında `<think>` bloğu içinde muhakemesini gösterir.

## Çıktı formatı

Model, eğitim verisindeki kalıbı üretir. Kullanırken bunu bilmen gerekir:

```
<think>
…muhakeme, ara hesap adımları…
</think>

<tool_call>{"name": "compute_derivative", "arguments": {"expression": "(3*x^2+1)*sin(x)"}}</tool_call>
```

Sen bu çağrıyı **gerçekten çalıştırır** ve sonucu **kullanıcı turu** olarak geri verirsin:

```
<tool_response>{"derivative": "6*x*sin(x) + (3*x^2+1)*cos(x)"}</tool_response>
```

Model ardından son yanıtı doğal dille üretir. Araç gerekmeyen sorularda `<tool_call>`
hiç üretilmez, `<think>` sonrası doğrudan cevap gelir.

## Kullanım

### Unsloth ile

```python
from unsloth import FastModel

model, tokenizer = FastModel.from_pretrained(
    model_name     = "bilalabic/gemma_4_math-toolcall-tr_lora",
    max_seq_length = 2048,
    load_in_4bit   = True,
)

messages = [{
    "role": "user",
    "content": [{"type": "text", "text":
        "Elimdeki 12 bin lira yılda %14 getiriyle 2 yıl sonra ne olur?"}],
}]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True,
    tokenize=True, return_tensors="pt", return_dict=True,
).to("cuda")

outputs = model.generate(**inputs, max_new_tokens=512,
                         temperature=1.0, top_p=0.95, top_k=64)
print(tokenizer.batch_decode(outputs)[0])
```

`temperature=1.0, top_p=0.95, top_k=64` Gemma-4 ekibinin önerdiği ayarlardır.

### PEFT ile

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("unsloth/gemma-4-e4b-it-unsloth-bnb-4bit")
model = PeftModel.from_pretrained(base, "bilalabic/gemma_4_math-toolcall-tr_lora")
tokenizer = AutoTokenizer.from_pretrained("bilalabic/gemma_4_math-toolcall-tr_lora")
```

### Araç döngüsü

```python
# 1) modelden yanit al -> <tool_call> ciktisini ayikla
# 2) fonksiyonu KENDI kodunla calistir
# 3) sonucu kullanici turu olarak geri besle:
messages.append({"role": "assistant", "content": [{"type": "text", "text": model_output}]})
messages.append({"role": "user", "content": [{"type": "text",
    "text": '<tool_response>{"amount": "15595.20"}</tool_response>'}]})
# 4) tekrar generate -> dogal dilde son cevap
```

## Eğitim

[Unsloth](https://github.com/unslothai/unsloth) ile Google Colab (A100) üzerinde,
`train_on_responses_only` kullanılarak — kayıp yalnızca model yanıtlarında hesaplanır,
kullanıcı ve `tool_response` turları maskelenir. Bu sayede model araç sonucunu
**uydurmayı değil, yorumlamayı** öğrenir.

📓 Eğitim notebook'u: [`notebooks/gemma4_e4b_math_toolcall_lora.ipynb`](https://github.com/BilalAbic/math-toolcall-tr/blob/main/notebooks/gemma4_e4b_math_toolcall_lora.ipynb)

### LoRA yapılandırması

| Ayar | Değer |
|---|---|
| `r` | 8 |
| `lora_alpha` | 8 |
| `lora_dropout` | 0 |
| `bias` | none |
| Hedef katmanlar | yalnızca dil katmanları (attention + MLP) |
| Görsel katmanlar | eğitilmedi |
| `task_type` | CAUSAL_LM |
| PEFT sürümü | 0.19.1 |
| Kuantizasyon | 4-bit (bnb) |

### Eğitim hiperparametreleri

| Ayar | Değer |
|---|---|
| Epoch | 3 |
| Toplam adım | 285 |
| Öğrenme oranı | 2e-4 |
| Batch | 2 × 4 (grad. accum.) = **8** |
| Warmup adımı | 5 |
| Optimizer | adamw_8bit |
| Weight decay | 0.001 |
| LR zamanlayıcı | linear |
| `max_seq_length` | 2048 |
| Seed | 3407 |

### Eğitim sonucu

| | |
|---|---|
| Eğitilen örnek | 757 |
| Eğitilebilir parametre | 18.350.080 / 8.014.506.528 (**%0,23**) |
| Son eğitim kaybı | ≈ 0,076 |
| Donanım | NVIDIA A100-SXM4-80GB (Colab) |
| Ayrılan bellek | ≈ 10,3 GB |

> ⚠️ **Sürüm farkı — önemli.** Bu adaptör, veri setinin **757 örneklik daha eski bir
> anlık görüntüsüyle** eğitilmiştir. Veri seti o tarihten sonra **2.127 örneğe**
> genişletilmiştir ve güncel sürümle **yeniden eğitim henüz yapılmamıştır**.
> Dolayısıyla buradaki ağırlıklar, veri setinin bugünkü hâlinin tamamını görmemiştir.

## Değerlendirme

Adaptör, aynı temel modelin LoRA uygulanmamış hâliyle üç benchmark üzerinde
karşılaştırılmıştır. Sonuçlar tek bir çalıştırmaya aittir; güven aralığı veya birden
fazla seed ortalaması değildir.

| Benchmark | Metrik | Base | Fine-tune | Fark |
|---|---|---:|---:|---:|
| Türkçe MMLU | Doğruluk | %75,20 (188/250) | %68,00 (170/250) | **−7,20 puan** |
| Matematik Tool-Call | Genel doğruluk | %56,67 | %59,33 | **+2,66 puan** |
| Matematik Tool-Call | Araç seçimi | %53,77 | %54,72 | **+0,95 puan** |
| Matematik Tool-Call | Çekimserlik (`abstain`) | %63,64 | %70,45 | **+6,81 puan** |
| Matematik Tool-Call | Format geçerliliği | %98,67 | %99,33 | **+0,66 puan** |
| GSM8K | Doğruluk | %66,00 (99/150) | %73,33 (110/150) | **+7,33 puan** |
| GSM8K | Gereksiz araç çağrısı (düşük daha iyi) | %0,00 | %14,00 | **+14,00 puan** |

### Test düzeni

- **Türkçe MMLU:** 250 çoktan seçmeli örnek; genel bilginin eğitim sonrasında korunup
  korunmadığını ölçer.
- **Matematik Tool-Call:** Eğitimde kullanılan ilk 757 kayıttan sonra gelen, modelin
  eğitim sırasında görmediği 571 kayıt arasından senaryo dengeli seçilmiş 150 örnek.
  Eğitim ve değerlendirme kümelerinin sırası korunmuş ve aralarında çakışma olmadığı
  doğrulanmıştır.
- **GSM8K:** Standart test bölümünden 150 İngilizce matematik problemi. Doğruluğun
  yanında, araç sunulmayan standart sorularda modelin kendiliğinden `<tool_call>`
  üretip üretmediği de ölçülmüştür.

### Sonuçların yorumu

Fine-tune, hedeflenen tool-calling görevinde sınırlı fakat tutarlı bir artış sağlamıştır.
En belirgin kazanım, araç çağırmaması gereken örneklerdeki çekimserliktir
(+6,81 puan). GSM8K doğruluğu da 7,33 puan yükselmiştir.

Bununla birlikte iki önemli gerileme vardır:

1. Türkçe MMLU doğruluğu 7,20 puan düşmüştür; bu sonuç genel bilgi üzerinde
   **catastrophic forgetting / alan daralması riski** bulunduğunu gösterir.
2. Fine-tune, araç verilmeyen GSM8K sorularının %14'ünde gereksiz `<tool_call>`
   üretmiştir. Bu, modelin eğitim formatına aşırı uyum gösterebildiğini belirtir.

Bu nedenle adaptör, genel amaçlı temel modelin doğrudan yerine geçen bir yükseltme
olarak değil, Türkçe matematik tool-calling için deneysel bir adaptör olarak
değerlendirilmelidir.

## Eğitim verisi

[`bilalabic/math-toolcall-tr`](https://huggingface.co/datasets/bilalabic/math-toolcall-tr)
— Türkçe matematik tool-calling veri seti; 13 alt alan, 70 konu, 8 araç çağırma senaryosu.

**Bu adaptör veri setinin 757 örneklik sürümüyle eğitilmiştir**; yayımlanan veri seti
şu anda 2.127 örnektir (yukarıdaki sürüm farkı notuna bakınız).

Veri kasıtlı olarak yalnızca "başarılı çağrı" içermez: örneklerin **%29,6'sında hiç
araç çağrılmaz** (soru zaten cevaplanabilir ya da zorunlu parametre eksik). Senaryolar
arasında `yanlis_arac_tuzagi` (benzer isimli araçlardan doğrusunu seçme), `hata_yonetimi`
(sıfıra bölme vb.), `zincirli_cagri` ve `paralel_cagri` bulunur.

## Kaynaklar

| | |
|---|---|
| 💻 GitHub (kod + notebook) | https://github.com/BilalAbic/math-toolcall-tr |
| 📓 Eğitim notebook'u | [gemma4_e4b_math_toolcall_lora.ipynb](https://github.com/BilalAbic/math-toolcall-tr/blob/main/notebooks/gemma4_e4b_math_toolcall_lora.ipynb) |
| 📊 Veri seti | https://huggingface.co/datasets/bilalabic/math-toolcall-tr |
| 🛠️ Veri üretim kodu | [toolcall-dataset/](https://github.com/BilalAbic/math-toolcall-tr/tree/main/toolcall-dataset) |

## Sınırlamalar

- **Eğitim verisi sentetiktir** ve insan doğrulamasından geçmemiştir; `tool_response`
  içerikleri gerçek fonksiyon çalıştırılarak değil, model tarafından üretilmiştir.
- **Hesap doğruluğu garanti değildir.** Model araç *çağırmayı* öğrenir, hesabı kendisi
  yapmaz — kritik sonuçlarda çağrıyı gerçekten çalıştır ve sonucu doğrula.
- **Yalnızca Türkçe ve yalnızca matematik.** Başka dil veya alanlara genellemesi beklenmez.
- **Küçük veri seti (757 örnek)** ve `r=8` — üslup ve format öğrenilmiştir, temel modelin
  matematik yeteneği köklü biçimde değişmemiştir.
- Benchmark yalnızca tek çalıştırma ve sınırlı örneklemle yapılmıştır; sonuçlar farklı
  seed, decoding ayarı veya donanım/yazılım sürümlerinde değişebilir.
- Türkçe MMLU'da **−7,20 puan** genel bilgi kaybı ve GSM8K'da **%14 gereksiz araç
  çağrısı** ölçülmüştür. Genel amaçlı veya araçsız kullanımda bu riskler dikkate
  alınmalıdır.
- Temel modelin ve `<tool_call>` formatının dışına çıkan araç şemalarında davranış
  öngörülemez olabilir.

## Lisans

Apache-2.0. Temel model [Gemma kullanım şartlarına](https://ai.google.dev/gemma/terms)
tabidir.

## Alıntı

```bibtex
@misc{gemma4-math-toolcall-tr-lora,
  title  = {gemma_4_math-toolcall-tr_lora: Turkish Math Tool-Calling LoRA for Gemma-4},
  author = {Bilal Abiç},
  year   = {2026},
  url    = {https://huggingface.co/bilalabic/gemma_4_math-toolcall-tr_lora}
}
```

[<img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20made%20with%20love.png" width="200"/>](https://github.com/unslothai/unsloth)
