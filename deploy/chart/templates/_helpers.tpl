{{- define "anvilkit-agent-inference.name" -}}
anvilkit-agent-inference
{{- end }}

{{- define "anvilkit-agent-inference.fullname" -}}
{{- if contains "anvilkit-agent-inference" .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-anvilkit-agent-inference" .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end }}

{{- define "anvilkit-agent-inference.labels" -}}
app.kubernetes.io/name: {{ include "anvilkit-agent-inference.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "anvilkit-agent-inference.selectorLabels" -}}
app.kubernetes.io/name: {{ include "anvilkit-agent-inference.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{- define "anvilkit-agent-inference.serviceAccountName" -}}
{{- default (include "anvilkit-agent-inference.fullname" .) .Values.serviceAccount.name -}}
{{- end }}

{{- define "anvilkit-agent-inference.image" -}}
{{- if .Values.image.digest -}}
{{- printf "%s@%s" .Values.image.repository .Values.image.digest -}}
{{- else -}}
{{- printf "%s:%s" .Values.image.repository (required "image.tag or image.digest is required" .Values.image.tag) -}}
{{- end -}}
{{- end }}
