from django.contrib import admin
from django.urls import path, include
from respose import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('video/', views.shipin),  # 引入 video_analysis 视图
]
