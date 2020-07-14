from django.shortcuts import render
from django.http import HttpResponse
from django.contrib.auth import get_user_model


UserModel = get_user_model()


def index(request):
    return HttpResponse("Hello, world. You're at ArClassifier index.")


def login(request):
    if request.user.is_authenticated :
        return redirect('/home')  
    else:        
        form=UserForm(request.POST)
        context = {'form':form}
        if request.method == 'POST':
            username = request.POST.get('username')
            password = request.POST.get('password')
            user = authenticate(username=username, password=password)
            if user:
                if user.is_active:
                    auth_login(request,user)
                    return redirect('home')
                else:
                    return HttpResponse("Your account is inactive.")
            else:
                errors = '<div class="alert alert-danger" role="alert">username or password are incorrect</div>'
                context = {'form':form, 'errors': errors}
                return render(request, 'login.html', context)
        else:
            return render(request, 'login.html', context)


def signup(request):
    if request.user.is_authenticated :
        return redirect('/home')
    else:
        if (request.method == 'POST'):
            form=UserRegistreForm(request.POST)
            if form.is_valid():
                user = form.save(commit=False)
                user.save()
                raw_password = request.POST.get('password1')
                raw_user=request.POST.get('username')
                user = authenticate(username=raw_user, password=raw_password)
                auth_login(request, user)
                return redirect('home')
            else:
                context = {'form':form}
                return render(request, 'signup.html', context)
        else:
            partial_token = None
            if request.GET.get('partial_token'):
                strategy = load_strategy()
                partial_token = request.GET.get('partial_token')
                partial = strategy.partial_load(partial_token)
                data = partial.data['kwargs']['details']
                form=UserSignUpForm(initial = {'username': data['username'],'email' :data['email']})
                context = {'form':form}
                return render(request, 'signup.html', context)
            else:
                form=UserSignUpForm()
                context = {'form':form}
                return render(request, 'signup.html', context)


def error_404(request,exception):
    context = {}
    return render(request, '404_page.html', context)


