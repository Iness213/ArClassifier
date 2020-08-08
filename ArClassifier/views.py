from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib.auth import get_user_model, authenticate, login, logout

from ArClassifier.form import LogInForm

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
