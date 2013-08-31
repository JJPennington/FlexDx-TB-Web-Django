from django.conf.urls import patterns, include, url
from django.conf import settings
from django.conf.urls.static import static

# Uncomment the next two lines to enable the admin:
# from django.contrib import admin
# admin.autodiscover()

urlpatterns = patterns('',
    # Examples:
    # url(r'^$', 'flexdxtb.views.home', name='home'),
    # url(r'^flexdxtb/', include('flexdxtb.foo.urls')),

    # Uncomment the admin/doc line below to enable admin documentation:
    # url(r'^admin/doc/', include('django.contrib.admindocs.urls')),

    # Uncomment the next line to enable the admin:
    # url(r'^admin/', include(admin.site.urls)),
    url(r'^$', 'async.views.home_page'),
    url(r'^index.html$', 'async.views.home_page'),
    url(r'^about/$', 'async.views.about_page'),
    url(r'^model/$', 'async.views.model_page'),
    url(r'^bargraph.png$', 'async.views.bargraph'),
    url(r'^dgraph1.png$', 'async.views.dgraph1'),
    url(r'^dgraph2.png$', 'async.views.dgraph2'),
) + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
