from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib.auth import get_user_model, authenticate, login, logout
from django.views.generic.edit import FormView
from ArClassifier.form import LogInForm, SignUpForm, ForgotPassForm

UserModel = get_user_model()


def index(request):
    context = {}
    return render(request, 'dashboard.html', context)


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
                new_user = authenticate(username=user.email, password=password)
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
