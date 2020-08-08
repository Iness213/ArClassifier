from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib.auth import get_user_model, authenticate, login, logout

from ArClassifier.form import LogInForm, SignUpForm, ForgotPassForm

UserModel = get_user_model()


def index(request):
    return HttpResponse("Hello, world. You're at ArClassifier index.")


def login_view(request):
    if request.user.is_authenticated:
        return redirect('/home')
    else:
        next = request.GET.get('next')
        form = LogInForm(request.POST or None)
        context = {'form': form}
        if request.method == 'POST':
            if form.is_valid():
                email = form.cleaned_data.get('login')
                password = form.cleaned_data.get('password')
                user = authenticate(email=email, password=password)
                login(request, user)
                if next:
                    return redirect(next)
                return redirect('/')

            context = {
                'form': form,
            }
        return render(request, 'login.html', context)


def signup_view(request):
    if request.user.is_authenticated:
        return redirect('/home')
    else:
        next = request.GET.get('next')
        form = SignUpForm(request.POST or None)
        context = {'form': form}
        if request.method == 'POST':
            if form.is_valid():
                user = form.save(commit=False)
                password = form.cleaned_data.get('password')
                user.set_password(password)
                user.save()
                new_user = authenticate(email=user.email, password=password)
                login(request, new_user)
                if next:
                    return redirect(next)
                return redirect('/')

            context = {
                'form': form,
            }
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
