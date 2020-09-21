import json
import os
from pathlib import Path
import ntpath
from django.contrib.auth import get_user_model, authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.core.serializers.json import DjangoJSONEncoder
from django.core.files.storage import default_storage
from django.http import HttpResponse
from django.shortcuts import render, redirect

from ArClassifier.form import LogInForm, SignUpForm, ForgotPassForm
from ArClassifier.models import MyUser, Project, Dataset, Keyword, Result, Classification, TrainingSet, Metric
from Text_Classification.settings import BASE_DIR

from .Preprocessing import finale_preprocess
from .SAprediction import predict_naive_bayes
from .predictor import predict, word_cloud

UserModel = get_user_model()


@login_required(login_url='login/')
def index(request):
    if 'user_email' in request.session:
        user_email = request.session['user_email']
        curernt_user = MyUser.objects.filter(email__exact=user_email).first()
        user_projects_qs = Project.objects.filter(owner=curernt_user)
        number_of_datasets = 0
        number_of_saved_results = 0
        for project in user_projects_qs:
            project_datasets_qs = Dataset.objects.filter(project=project)
            number_of_datasets = number_of_datasets + len(project_datasets_qs)
            project_classification_qs = Classification.objects.filter(project=project)
            number_of_saved_results = number_of_saved_results + len(project_classification_qs)

        context = {
            'number_of_datasets': number_of_datasets,
            'number_of_saved_results': number_of_saved_results,
        }
        return render(request, 'dashboard.html', context)
    else:
        redirect('/login/')


@login_required(login_url='login/')
def projects(request):
    user_email = request.session['user_email']
    curernt_user = MyUser.objects.filter(email__exact=user_email).first()
    user_projects_qs = Project.objects.filter(owner=curernt_user)
    qs = user_projects_qs
    l = []
    for p in qs:
        l.append(p.name)
    u = json.dumps(l, cls=DjangoJSONEncoder)
    context = {
        'projects': user_projects_qs,
        'projects_json': u
    }
    return render(request, 'projects.html', context)


@login_required(login_url='login/')
def project(request, id):
    prjct = Project.objects.get(id=id)
    files = Dataset.objects.filter(project=prjct)
    classifications = Classification.objects.filter(project=prjct)
    hasFiles = False
    hasClassification = False
    if len(files) > 0:
        hasFiles = True
    if len(classifications) > 0:
        hasClassification = True
    context = {
        'project': prjct,
        'hasFiles': hasFiles,
        'hasClassification': hasClassification,
        'files': files,
        'classifications': classifications
    }
    return render(request, 'project.html', context)


@login_required(login_url='login/')
def add_project(request):
    if request.method == 'POST':
        user_email = request.session['user_email']
        curernt_user = MyUser.objects.filter(email__exact=user_email).first()
        name = request.POST.get('name')
        description = request.POST.get('description')
        new_project = Project(name=name, description=description, owner=curernt_user)
        new_project.save()
        return redirect('/project/' + str(new_project.id))


@login_required(login_url='login/')
def delete_project(request, id):
    project_to_delete = Project.objects.get(id=id)
    project_to_delete.delete()
    return redirect('/projects/')


@login_required(login_url='login/')
def file(request, id):
    file = Dataset.objects.get(id=id)
    project = Project.objects.get(id=file.project.id)
    name = file.name
    path = file.path
    with (open(path, 'r', encoding='UTF-8')) as reader:
        file_content = reader.read()
        reader.close()
    context = {
        'file_name': name,
        'file_content': file_content,
        'project_id': project.id,
        'project_name': project.name
    }
    return render(request, 'file.html', context)


@login_required(login_url='login/')
def feed_text(request, id):
    project = Project.objects.get(id=id)
    if request.method == 'POST':
        name = request.POST.get('name')
        text = request.POST.get('text')
        user_email = request.session['user_email']
        curernt_user = MyUser.objects.filter(email__exact=user_email).first()
        Path(BASE_DIR+'/files/'+str(curernt_user.id)).mkdir(parents=True, exist_ok=True)
        path = BASE_DIR+'/files/'+str(curernt_user.id)+'/'+name+'.txt'
        f = open(path, 'w+', encoding='UTF-8')
        f.write(text)
        d = Dataset(name=name+'.txt', project=project, path=path)
        d.save()
        return redirect('/project/' + str(id))

    context = {
        'projectId': project.id,
        'project_name': project.name
    }
    return render(request, 'feedText.html', context)


@login_required(login_url='login/')
def upload_file(request, id):
    if request.method == 'POST':
        project = Project.objects.get(id=id)
        user_email = request.session['user_email']
        curernt_user = MyUser.objects.filter(email__exact=user_email).first()
        file = request.FILES['file']
        path = BASE_DIR+'/files/'+str(curernt_user.id)+'/'+file.name
        file_name = default_storage.save(path, file)
        dataset = Dataset(name=file.name, path=path, project=project)
        dataset.save()
    context = {
        'projectId': id
    }
    return render(request, 'uploadFile.html', context)


@login_required(login_url='login/')
def delete_file(request, id):
    file_to_delete = Dataset.objects.get(id=id)
    project_id = file_to_delete.project.id
    file_to_delete.delete()
    return redirect('/project/' + str(project_id))


@login_required(login_url='login/')
def download_file(request, id):
    file = Dataset.objects.get(id=id)
    path = file.path
    with(open(path, 'r', encoding="UTF-8")) as reader:
        file_content = reader.read()
        response = HttpResponse(file_content, content_type='application/octet-stream')
        response['Content-Disposition'] = 'attachment; filename=' + file.name
        reader.close()
        return response


@login_required(login_url='login/')
def preprocess_text(request, id):
    file = Dataset.objects.get(id=id)
    path = file.path
    with(open(path, 'r', encoding='UTF-8')) as reader:
        file_content = reader.read()
        reader.close()
    pre_processed_text = finale_preprocess(file_content)
    context = {
        'project': file.project,
        'pre_processed_text': pre_processed_text,
        'file_content': file_content,
        'file': file
    }
    return render(request, 'preprocess.html', context)


@login_required(login_url='login/')
def sentiment(request, id):
    project = Project.objects.get(id=id)
    files = Dataset.objects.filter(project_id=id)
    if request.method == 'POST':
        dataset = TrainingSet.objects.get(id=2)
        file_id = request.POST.get('file')
        f = Dataset.objects.get(id=file_id)
        with (open(f.path, 'r', encoding='UTF-8')) as reader:
            file_content = reader.read()
            reader.close()
        prediction_class = predict_naive_bayes(file_content)
        c = Classification(training_set=dataset, file=f, project=project, type=prediction_class)
        c.save()
        return redirect('/project/' + str(id))
    context = {
        'project': project,
        'files': files,
    }
    return render(request, 'sentiment.html', context)


@login_required(login_url='login/')
def classification(request, id):
    project = Project.objects.get(id=id)
    datasets = TrainingSet.objects.all()
    files = Dataset.objects.filter(project_id=id)
    classifications = Classification.objects.filter(project=project)
    algorithms = {'KNN', 'SVM', 'Naive Bayes'}
    if request.method == 'POST':
        user_email = request.session['user_email']
        current_user = MyUser.objects.filter(email__exact=user_email).first()
        dataset = 1
        dataset = TrainingSet.objects.get(id=dataset)
        file_id = request.POST.get('file')
        f = Dataset.objects.get(id=file_id)
        with (open(f.path, 'r', encoding='UTF-8')) as reader:
            file_content = reader.read()
            reader.close()
        algorithm = request.POST.get('algorithm')
        algorithm = str(algorithm).replace(' ', '_')
        file_name = ntpath.basename(f.path)
        file_name = file_name.split('.')[0]+'.png'
        if algorithm == 'KNN':
            k_value = 5
            category, keywords = predict(file_content, algorithm, current_user.id, file_name, 'nada.csv', k_value)
        else:
            category, keywords = predict(file_content, algorithm, current_user.id, file_name)
        algorithm = str(algorithm).replace('_', ' ')
        if algorithm == 'KNN':
            c = Classification(training_set=dataset, file=f, project=project, algorithm=algorithm, k_value=5)
        else:
            c = Classification(training_set=dataset, file=f, project=project, algorithm=algorithm)
        c.save()
        r = Result(category=category, classification=c)
        r.save()
        for keyword in keywords:
            k = Keyword(word=keyword, result=r)
            k.save()
        return redirect('/result/' + str(c.id))
    context = {
        'project': project,
        'datasets': datasets,
        'files': files,
        'algorithms': algorithms,
    }
    return render(request, 'classification.html', context)


@login_required(login_url='login/')
def result(request, id):
    classification = Classification.objects.get(id=id)
    user_email = request.session['user_email']
    current_user = MyUser.objects.filter(email__exact=user_email).first()
    f = classification.file.path
    file_name = ntpath.basename(f)
    file_name = file_name.split('.')[0]+'.png'
    image_src = os.path.join(os.path.join('graph', str(current_user.id)), file_name)

    result = Result.objects.get(classification_id=classification.id)
    keywords = Keyword.objects.filter(result=result).first()
    project = classification.project
    metric = Metric.objects.filter(algorithm=classification.algorithm).first()
    word_cloud_src = word_cloud(keywords.word, result.id)
    if word_cloud_src is not None:
        context = {
            'project': project,
            'result': result,
            'classification': classification,
            'keywords': keywords,
            'metric': metric,
            'image_src': image_src,
            'word_cloud_src': word_cloud_src
        }
    else:
        context = {
            'project': project,
            'result': result,
            'classification': classification,
            'keywords': keywords,
            'metric': metric,
            'image_src': image_src
        }
    return render(request, 'result.html', context)


def login_view(request):
    if request.user.is_authenticated:
        return redirect('/home')
    else:
        if request.method == 'POST':
            form = LogInForm(request.POST)
            if form.is_valid():
                next = request.GET.get('next')
                email = form.cleaned_data.get('email')
                password = form.cleaned_data.get('password')
                user = authenticate(username=email, password=password)
                login(request, user)
                request.session['user_email'] = user.email
                if next:
                    return redirect(next)
                return redirect('/')
            else:
                i = 20
        else:
            form = LogInForm(request.POST or None)
            context = {'form': form}
            return render(request, 'login.html', context)


def signup_view(request):
    if request.user.is_authenticated:
        return redirect('/home')
    else:
        if request.method == 'POST':
            form = SignUpForm(request.POST)
            if form.is_valid():
                next = request.GET.get('next')
                user = form.save(commit=False)
                password = form.cleaned_data.get('password')
                user.set_password(password)
                user.username = user.email
                user.save()
                login(request, user)
                request.session['user_email'] = user.email
                if next:
                    return redirect(next)
                return redirect('/login/')
            else:
                i = 20
        else:
            form = SignUpForm()
            context = {'form': form}
            return render(request, 'signup.html', context)


@login_required(login_url='login/')
def logout_view(request):
    logout(request)
    return redirect('/login/')


def forgotpass_view(request):
    if request.user.is_authenticated:
        return redirect('/home')
    else:
        next = request.GET.get('next')
        form = ForgotPassForm(request.POST or None)
        context = {'form': form}
        if request.method == 'POST':
            if form.is_valid():
                ###################################################
                # YOU NEED TO SEND EMAIL HERE
                ###################################################
                if next:
                    return redirect(next)
                return redirect('/')

            context = {
                'form': form,
            }
        return render(request, 'forgotpass.html', context)
