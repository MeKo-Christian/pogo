package version

// Build-time variables set by ldflags
var (
	Version   = "dev"
	GitCommit = "unknown"
	BuildDate = "unknown"
)

// Info returns version information
func Info() (string, string, string) {
	return Version, GitCommit, BuildDate
}