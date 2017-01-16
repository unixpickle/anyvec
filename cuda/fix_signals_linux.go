//+build !nocuda, linux

package cuda

/*
#include <unistd.h>
#include <signal.h>

void anyvec_cuda_fix_signal(int n) {
	struct sigaction a;
	if (sigaction(n, NULL, &a) >= 0) {
		a.sa_flags |= SA_ONSTACK;
		sigaction(n, &a, NULL);
	}
}

void anyvec_cuda_fix_signals() {
	int signals[] = {SIGCHLD, SIGHUP, SIGINT, SIGQUIT, SIGABRT, SIGFPE, SIGTERM, SIGBUS,
		SIGSEGV, SIGXCPU, SIGXFSZ};
	int numSigs = sizeof(signals) / sizeof(int);
	for (int i = 0; i < numSigs; ++i) {
		anyvec_cuda_fix_signal(signals[i]);
	}
}
*/
import "C"

func fixSignals() {
	C.anyvec_cuda_fix_signals()
}
