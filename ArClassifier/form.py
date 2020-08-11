from django.contrib.auth import authenticate, get_user_model
from django import forms

User = get_user_model()


class LogInForm(forms.Form):
    email = forms.EmailField(widget=forms.EmailInput(attrs={
        'class': 'form-control form-control-lg',
        'placeholder': 'Enter your E-mail',
    }))
    password = forms.CharField(widget=forms.PasswordInput(attrs={
        'class': 'form-control form-control-lg',
        'placeholder': 'Enter your password',
    }))

    def clean(self, *args, **kwargs):
        email = self.cleaned_data.get('email')
        password = self.cleaned_data.get('password')

        if email and password:
            user = authenticate(username=email, password=password)
            if not user:
                raise forms.ValidationError('This user does not exist')
            if not user.check_password(password):
                raise forms.ValidationError('Incorrect password')
            if not user.is_active:
                raise forms.ValidationError('This user is not active')
            if user.is_banned:
                raise forms.ValidationError('Your account has been banned.')
        return super(LogInForm, self).clean()


class SignUpForm(forms.ModelForm):
    email = forms.CharField(widget=forms.EmailInput(attrs={
        'class': 'form-control form-control-lg',
        'placeholder': 'Enter your E-mail',
    }))
    password = forms.CharField(widget=forms.PasswordInput(attrs={
        'class': 'form-control form-control-lg',
        'placeholder': 'Enter your password',
    }))
    password1 = forms.CharField(widget=forms.PasswordInput(attrs={
        'class': 'form-control form-control-lg',
        'placeholder': 'Confirm your password',
    }))

    class Meta:
        model = User
        fields = [
            'email',
            'password',
            'password1'
        ]

    def clean(self):
        email = self.cleaned_data.get('email')
        password = self.cleaned_data.get('password')
        password1 = self.cleaned_data.get('password1')

        if password != password1:
            raise forms.ValidationError('Password must match')

        email_qs = User.objects.filter(email=email)

        if email_qs.exists():
            raise forms.ValidationError('This email has already been registered')
        return super(SignUpForm, self).clean()


class ForgotPassForm(forms.Form):
    email = forms.EmailField(widget=forms.EmailInput(attrs={
        'class': 'form-control form-control-lg',
        'placeholder': 'Enter your E-mail',
    }))

    def clean(self):
        email = self.cleaned_data.get('email')
        email_qs = User.objects.filter(email=email)
        if not email_qs.exists():
            raise forms.ValidationError('This is not a registered email')
        return super(ForgotPassForm, self).clean()
