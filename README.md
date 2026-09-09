# OCR-for-handwriting-improvement
**Some helper functions to improve the digitalisation of handwritten documents using standard OCR.**

OCR struggles with handwriting fonts, which is one of the obstacles when introducing fuly automatic digitalisation into real world practice. 

Here are some helper functions potentially benefitial to OCR performance with certain level of contextualisation and fine-tuning.

#### What is done
1. Basic image preprocessing (relies on a provided JSON file of the document structure)
2. Image quality check
3. A training script for a CNN model (MobileNetV2) to perform symbol detection and classification for local documenting rules or personal writing habits (requires training data)
4. A script using llama 3.1 for attempt to fix the raw OCR reading by providing the model information about the context.
5. A benchmark script to compare our solution with the industrial standard solution Google Cloud Vision

#### To do
1. UI
2. SSL