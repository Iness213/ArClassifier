class UserSingUpForm(UserCreationForm):
    email=forms.EmailField()
   
    class Meta:
        model=MyUser
        fields=['username','email','password1','password2']


class UserForm(forms.ModelForm):
    password = forms.CharField(widget=forms.PasswordInput())
    class Meta():
        model = MyUser
        fields = ['username','password']  