from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django.contrib.auth import get_user_model, authenticate, login, logout
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
            number_of_datasets = number_of_datasets + project_datasets_qs.len()

            for dataset in project_datasets_qs:
                dataset_keywords_qs = Keyword.objects.filter(dataset=dataset)
                number_of_keywords = number_of_keywords + dataset_keywords_qs.len()

                for keyword in dataset_keywords_qs:
                    keyword_results_qs = Result.objects.filter(keyword=keyword)
                    number_of_saved_results = number_of_saved_results + keyword_results_qs.len()
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
    context = {}
    return render(request, 'projects.html', context)


@login_required(login_url='login/')
def datasets(request):
    context = {}
    return render(request, 'datasets.html', context)


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
