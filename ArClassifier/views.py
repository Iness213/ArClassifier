from django.contrib.auth import get_user_model, authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect

from ArClassifier.form import LogInForm, SignUpForm, ForgotPassForm
from ArClassifier.models import MyUser, Project, Dataset, Keyword, Result

UserModel = get_user_model()


@login_required(login_url='login/')
def index(request):
    if 'user_email' in request.session:
        user_email = request.session['user_email']
        curernt_user = MyUser.objects.filter(email__exact=user_email).first()
        user_projects_qs = Project.objects.filter(owner=curernt_user)
        number_of_datasets = 0
        number_of_saved_results = 0
        number_of_keywords = 0
        for project in user_projects_qs:
            project_datasets_qs = Dataset.objects.filter(project=project)
            number_of_datasets = number_of_datasets + len(project_datasets_qs)

            for dataset in project_datasets_qs:
                dataset_keywords_qs = Keyword.objects.filter(dataset=dataset)
                number_of_keywords = number_of_keywords + len(dataset_keywords_qs)

                for keyword in dataset_keywords_qs:
                    keyword_results_qs = Result.objects.filter(keyword=keyword)
                    number_of_saved_results = number_of_saved_results + len(keyword_results_qs)
        context = {
            'number_of_datasets': number_of_datasets,
            'number_of_saved_results': number_of_saved_results,
            'number_of_keywords': number_of_keywords,
        }
        return render(request, 'dashboard.html', context)
    else:
        redirect('login/')


@login_required(login_url='login/')
def projects(request):
    user_email = request.session['user_email']
    curernt_user = MyUser.objects.filter(email__exact=user_email).first()
    user_projects_qs = Project.objects.filter(owner=curernt_user)
    context = {
        'projects': user_projects_qs
    }
    return render(request, 'projects.html', context)


@login_required(login_url='login/')
def project(request, id):
    prjct = Project.objects.get(id=id)
    files = Dataset.objects.filter(project=prjct)
    hasFiles = False
    if len(files) > 0:
        hasFiles = True
    context = {
        'project': prjct,
        'hasFiles': hasFiles,
        'files': files
    }
    return render(request, 'project.html', context)


@login_required(login_url='login/')
def datasets(request):
    context = {}
    return render(request, 'datasets.html', context)


@login_required(login_url='login/')
def feed_text(request):
    context = {}


@login_required(login_url='login/')
def upload_file(request):
    return None


@login_required(login_url='login/')
def classification(request):
    context = {}
    return render(request, 'classification.html', context)


@login_required(login_url='login/')
def saved_results(request):
    context = {}
    return render(request, 'savedResults.html', context)


@login_required(login_url='login/')
def help(request):
    context = {}
    return render(request, 'help.html', context)


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
                if next:
                    return redirect(next)
                return redirect('/')
            else:
                i = 20
        else:
            form = SignUpForm()
            context = {'form': form}
            return render(request, 'signup.html', context)


@login_required(login_url='login/')
def logout_view(request):
    logout(request)
    return redirect('/')


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


def add_project(request):
    if request.method == 'POST':
        user_email = request.session['user_email']
        curernt_user = MyUser.objects.filter(email__exact=user_email).first()
        name = request.POST.get('name')
        description = request.POST.get('description')
        new_project = Project(name=name, description=description, owner=curernt_user)
        new_project.save()
        return redirect('/project/' + str(new_project.id))


def delete_project(request, id):
    project_to_delete = Project.objects.get(id=id)
    project_to_delete.delete()
    return redirect('/projects/')


def delete_file(request, id):
    file_to_delete = Dataset.objects.get(id=id)
    project_id = file_to_delete.project.id
    file_to_delete.delete()
    return redirect('/project/' + str(project_id))
