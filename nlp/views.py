from django.shortcuts import render
from nlp.models import Dataset

# Create your views here.
def index(request):
    context = {
        "info": Dataset.info.to_html(index=False),
        "score_f1": Dataset.score_f1,
        "accuracy": Dataset.accuracy
    }
    return render(request, "nlp/index.html", context)

def dataset(request):
    context = {
        "info": Dataset.info.to_html(index=False),
        "dataset": Dataset.dataset.to_html(),
        "total_num": Dataset.total_num,
        "hate_num": Dataset.hate_num,
        "not_hate_num": Dataset.not_hate_num,
        "not_user_dataset": Dataset.not_user_dataset.to_html(),
        "clean_dataset": Dataset.clean_dataset.to_html(),
        "stemming_dataset": Dataset.stemming_dataset.to_html()
    }
    return render(request, "nlp/dataset.html", context)

def examples(request):
    if request.method == "POST":
        if 'from_data' in request.POST:
            values = Dataset.example()
            context = {
                "table_no": 1,
                "example_tweet": values[0],
                "example_sim_rac": values[1],
                "example_sim_sex": values[2],
                "hate_type": values[3]
            }
            return render(request, "nlp/examples.html", context)
        if 'from_user' in request.POST:
            tweet = request.POST["tweet"]

            values = Dataset.user_example(tweet)
            context = {
                "table_no": 2,
                "user_pred": values[0],
                "user_tweet": values[1],
                "user_example_tweet": values[2],
                "user_example_sim_rac": values[3],
                "user_example_sim_sex": values[4],
                "user_hate_type": values[5]
            }
            return render(request, "nlp/examples.html", context)
    return render(request, "nlp/examples.html")