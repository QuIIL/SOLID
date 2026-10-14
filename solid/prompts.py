"""Class names and rank prompts. Classes are listed in their ordinal (rank) order; split labels index these lists."""

CLASSES = {
    'panda': ['Benign', 'ISUP Grade 1', 'ISUP Grade 2', 'ISUP Grade 3', 'ISUP Grade 4', 'ISUP Grade 5'],
    'bracs': ['Normal', 'Pathological Benign', 'Usual Ductal Hyperplasia', 'Flat Epithelial Atypia',
              'Atypical Ductal Hyperplasia', 'Ductal Carcinoma in Situ', 'Invasive Carcinoma'],
    'rcc': ['Normal', 'Renal Oncocytoma', 'Chromophobe RCC', 'Papillary RCC', 'Clear Cell RCC'],
    'nsclc': ['Lung Adenocarcinoma', 'Lung Squamous Cell Carcinoma'],
}

PHRASES = {
    'panda': ['benign prostate tissue', 'ISUP Grade 1 prostate cancer', 'ISUP Grade 2 prostate cancer',
              'ISUP Grade 3 prostate cancer', 'ISUP Grade 4 prostate cancer', 'ISUP Grade 5 prostate cancer'],
    'bracs': ['normal breast tissue', 'pathological benign lesion in breast tissue',
              'usual ductal hyperplasia in breast tissue', 'flat epithelial atypia in breast tissue',
              'atypical ductal hyperplasia in breast tissue', 'ductal carcinoma in situ in breast tissue',
              'invasive carcinoma in breast tissue'],
    'rcc': ['normal kidney tissue', 'renal oncocytoma', 'chromophobe renal cell carcinoma',
            'papillary renal cell carcinoma', 'clear cell renal cell carcinoma'],
    'nsclc': ['lung adenocarcinoma', 'lung squamous cell carcinoma'],
}

RANK_PROMPTS = {name: ['A WSI showing a diagnosis of ' + p for p in phrases] for name, phrases in PHRASES.items()}


def get_prompts(dataset):
    return RANK_PROMPTS[dataset]
