from django.contrib import admin
from django.urls import path, include
from respose import views
from pushup import views as pu

urlpatterns = [
    path('admin/', admin.site.urls),
    path('squats/', views.shipin),  # 引入 video_analysis 视图
    path('pushup/', pu.shipin),  # 引入 video_analysis 视图
]
