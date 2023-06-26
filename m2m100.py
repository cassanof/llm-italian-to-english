from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

tokenizer.src_lang = "it"

while True:
    it_text = input("Enter some text in Italian: ")
    encoded_it = tokenizer(it_text, return_tensors="pt")
    generated_tokens = model.generate(
        **encoded_it, forced_bos_token_id=tokenizer.get_lang_id("en"))
    print(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))
    "La vie est comme une boîte de chocolat."
