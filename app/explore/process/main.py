from explore.models import Request


class Main:

    def run(self):
        el = Request.objects.using('mysql').first().relation_set.all()
        return el