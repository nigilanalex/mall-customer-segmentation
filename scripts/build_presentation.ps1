# Build an editable 15-slide presentation with installed Microsoft PowerPoint.
# Run: powershell -ExecutionPolicy Bypass -File scripts/build_presentation.ps1
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$doc = Join-Path $root 'documentation'
$details = Get-Content -LiteralPath (Join-Path $doc 'submission_details.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$output = Join-Path $doc 'AI_Customer_Segmentation_Presentation.pptx'
$render = Join-Path $doc '.qa/slides'
New-Item -ItemType Directory -Force -Path $render | Out-Null
$wasRunning = @(Get-Process POWERPNT -ErrorAction SilentlyContinue).Count -gt 0
$powerpoint = New-Object -ComObject PowerPoint.Application
$powerpoint.DisplayAlerts = 1
$deck = $powerpoint.Presentations.Add(0)
$deck.PageSetup.SlideWidth = 960
$deck.PageSetup.SlideHeight = 540

function Color([string]$hex) {
    $h = $hex.TrimStart('#')
    return [Convert]::ToInt32($h.Substring(0,2),16) + 256*[Convert]::ToInt32($h.Substring(2,2),16) + 65536*[Convert]::ToInt32($h.Substring(4,2),16)
}
$ink = Color '#173342'; $teal = Color '#0d9488'; $muted = Color '#516978'; $white = Color '#ffffff'
function Add-Text($slide,[string]$text,[double]$x,[double]$y,[double]$w,[double]$h,[double]$size=22,[int]$color=$ink,[bool]$bold=$false) {
    $shape = $slide.Shapes.AddTextbox(1,$x,$y,$w,$h)
    $shape.TextFrame.MarginLeft=0; $shape.TextFrame.MarginRight=0; $shape.TextFrame.MarginTop=0; $shape.TextFrame.MarginBottom=0
    $shape.TextFrame.WordWrap=-1
    $shape.TextFrame.TextRange.Text=$text
    $shape.TextFrame.TextRange.Font.Name='Arial'
    $shape.TextFrame.TextRange.Font.Size=$size
    $shape.TextFrame.TextRange.Font.Color.RGB=$color
    $shape.TextFrame.TextRange.Font.Bold= $(if($bold){-1}else{0})
    $shape.TextFrame.TextRange.ParagraphFormat.SpaceAfter=10
    return $shape
}
function New-Slide([string]$title) {
    $slide=$deck.Slides.Add($deck.Slides.Count+1,12)
    $slide.FollowMasterBackground=0
    $slide.Background.Fill.ForeColor.RGB=$white
    [void](Add-Text $slide $title 48 30 864 60 32 $ink $true)
    [void](Add-Text $slide ([string]$slide.SlideIndex) 900 509 25 16 10 $muted)
    return $slide
}
function Add-Body($slide,[string[]]$lines) {
    [void](Add-Text $slide ($lines -join "`r`n`r`n") 52 130 850 350 23 $ink)
}
function Add-Note($slide,[string]$text) {
    $slide.NotesPage.Shapes.Placeholders.Item(2).TextFrame.TextRange.Text=$text
}
function Add-PictureFit($slide,[string]$path,[double]$x,[double]$y,[double]$w,[double]$h) {
    $picture=$slide.Shapes.AddPicture($path,0,-1,$x,$y,-1,-1)
    $ratio=[Math]::Min($w/$picture.Width,$h/$picture.Height)
    $picture.LockAspectRatio=-1
    $picture.Width=$picture.Width*$ratio
    $picture.Left=$x+($w-$picture.Width)/2
    $picture.Top=$y+($h-$picture.Height)/2
}
function Add-Table($slide,$rows,[double[]]$widths,[double]$y=145,[double]$height=275,[double]$font=18) {
    $total=($widths|Measure-Object -Sum).Sum
    $shape=$slide.Shapes.AddTable($rows.Count,$widths.Count,48,$y,$total,$height)
    for($col=1;$col -le $widths.Count;$col++){$shape.Table.Columns.Item($col).Width=$widths[$col-1]}
    for($r=1;$r -le $rows.Count;$r++) {
        for($c=1;$c -le $widths.Count;$c++) {
            $cell=$shape.Table.Cell($r,$c).Shape
            $cell.TextFrame.TextRange.Text=[string]$rows[$r-1][$c-1]
            $cell.TextFrame.TextRange.Font.Name='Arial'
            $cell.TextFrame.TextRange.Font.Size=$font
            $cell.TextFrame.TextRange.Font.Color.RGB=$ink
            $cell.TextFrame.MarginLeft=9; $cell.TextFrame.MarginRight=7
            $cell.Fill.ForeColor.RGB=$(if($r -eq 1){Color '#d8eeea'}elseif($r%2 -eq 0){Color '#f3f7f9'}else{$white})
            if($r -eq 1){$cell.TextFrame.TextRange.Font.Bold=-1}
        }
    }
}

try {
    $s=New-Slide 'AI-Based Customer Segmentation'
    $s.Background.Fill.ForeColor.RGB=$ink
    $s.Shapes.Item(1).TextFrame.TextRange.Font.Color.RGB=$white
    $s.Shapes.Item(1).TextFrame.TextRange.Font.Size=40
    $s.Shapes.Item(1).Height=110
    [void](Add-Text $s "Personalized Marketing Recommendation System`r`nusing Machine Learning" 48 148 864 108 31 $white $true)
    [void](Add-Text $s "$($details.name)  |  $($details.roll_number)`r`n$($details.degree)  |  $($details.college)`r`nAcademic year: $($details.academic_year)`r`nGuide: $($details.guide)" 48 309 864 156 19 (Color '#c6e7e1'))
    Add-Note $s 'Editable title slide. Details supplied by the student; no guide or approval is invented. Synthetic demonstration data.'

    $s=New-Slide 'Introduction'
    Add-Body $s @('Customer behavior varies beyond income and spending score.','Purchase recency, frequency and monetary value add useful context.','This system combines unsupervised learning with transparent marketing rules.','The demonstration uses synthetic records and makes no claim of real campaign lift.')
    Add-Note $s 'Explain the distinction between learned groups and designed recommendation rules. Source: project README and src modules.'

    $s=New-Slide 'Problem statement'
    Add-Body $s @('How can an analyst turn customer and purchase records into interpretable customer groups?','Compare alternative clustering assumptions using a consistent feature space.','Explain new-customer assignments and propose actions without claiming ground-truth customer labels.')

    $s=New-Slide 'Objectives'
    Add-Body $s @('1. Derive and reconcile RFM from transaction records.','2. Compare four models using three internal metrics and coverage.','3. Explain customer assignments and segment-based marketing rules.','4. Deliver authenticated analytics, saved models and SQLite history.')

    $s=New-Slide 'Existing system'
    Add-PictureFit $s (Join-Path $root 'legacy/outputs/customer_clusters.png') 45 119 520 350
    [void](Add-Text $s "Original baseline`r`n`r`n500 synthetic customers`r`nIncome + spending score`r`nK-Means and elbow analysis`r`nSingle-page Streamlit app" 610 139 300 316 22 $ink)
    Add-Note $s 'Image is the original project output, preserved under legacy. The final application retains classic mode and legacy files.'

    $s=New-Slide 'Proposed system'
    Add-Body $s @('2,000 customers and 32,800 purchases with transaction-derived RFM.','Four clustering algorithms, comparison scores and explained assignments.','Administrator login, six protected views and local analysis history.','Segment profiles produce campaign suggestions that users can inspect and export.')

    $s=New-Slide 'System architecture'
    $labels=@('Admin login','Protected dashboard','CSV + RFM','SQLite snapshot','Four ML models','Profiles + marketing')
    $positions=@(@(48,150),@(360,150),@(672,150),@(672,320),@(360,320),@(48,320))
    for($i=0;$i -lt 6;$i++) {
        $x=$positions[$i][0]; $y=$positions[$i][1]
        $shape=$s.Shapes.AddShape(1,$x,$y,240,75)
        $shape.Fill.ForeColor.RGB=Color '#e8f5f2'; $shape.Line.ForeColor.RGB=$teal
        [void](Add-Text $s $labels[$i] ($x+15) ($y+23) 210 43 21 $ink $true)
        if($i -lt 2){$arrow=$s.Shapes.AddLine($x+240,$y+37,$x+302,$y+37)}
        elseif($i -eq 2){$arrow=$s.Shapes.AddLine($x+120,$y+75,$x+120,$y+160)}
        elseif($i -lt 5){$arrow=$s.Shapes.AddLine($x,$y+37,$x-62,$y+37)}
        else{$arrow=$null}
        if($null -ne $arrow){$arrow.Line.ForeColor.RGB=$teal;$arrow.Line.EndArrowheadStyle=3;$arrow.Line.Weight=2}
    }
    [void](Add-Text $s 'Local files and saved model bundles support reproducible analysis.' 48 446 864 36 20 $muted)

    $s=New-Slide 'Dataset'
    $rows=@(@('Component','Size / unit','Purpose'),@('Customers','2,000 records','Synthetic customer profiles'),@('Transactions','32,800 purchases','Auditable RFM aggregation'),@('Income / monetary','Thousands USD / USD','Explicit unit distinction'),@('Analysis date','2026-09-25','Reproducible recency'),@('Window','365 days, inclusive boundaries','Comparable observation period'))
    Add-Table $s $rows @(245,220,399) 130 305 18
    [void](Add-Text $s 'Gender and satisfaction are descriptive; IDs never enter model distances.' 48 458 864 40 19 $muted)

    $s=New-Slide 'RFM analysis'
    $rows=@(@('Measure','Definition','Worked example'),@('Recency','Days since latest purchase','5 days'),@('Frequency','Number of in-window purchases','3 purchases'),@('Monetary','Sum of in-window amounts','$300'))
    Add-Table $s $rows @(190,424,250) 139 235 21
    [void](Add-Text $s 'Example: $120 + $80 + $100, latest purchase 20 September, reference date 25 September.' 48 401 864 63 21 $ink)
    Add-Note $s 'Worked example is illustrative, not a claim about a particular dataset row. No-purchase customers use frequency/value 0 and recency sentinel 366.'

    $s=New-Slide 'Machine learning algorithms'
    $rows=@(@('Model','Training idea','New-customer assignment'),@('K-Means','Nearest mean, squared distance','Native prediction'),@('Ward hierarchy','Merge by variance increase','Nearest-centroid proxy'),@('DBSCAN','Density and noise','Core-sample radius extension'),@('Gaussian Mixture','Probabilistic components','Native posterior assignment'))
    Add-Table $s $rows @(220,310,334) 132 291 18
    [void](Add-Text $s 'Shared preprocessing: median imputation, log1p(RFM), standard scaling.' 48 450 864 44 20 $muted)
    Add-Note $s 'Sources: https://scikit-learn.org/stable/modules/clustering.html and https://scikit-learn.org/stable/modules/mixture.html. The extension rules are project implementation choices.'

    $s=New-Slide 'Model comparison'
    $rows=,@('Model','Groups','Coverage','Silhouette','DBI','CH')
    foreach($row in (Import-Csv (Join-Path $root 'outputs/model_comparison.csv'))) {
        $rows+=,@($row.Model,$row.Clusters,('{0:P1}' -f [double]$row.Coverage),('{0:F3}' -f [double]$row.Silhouette),('{0:F3}' -f [double]$row.'Davies-Bouldin'),('{0:F1}' -f [double]$row.'Calinski-Harabasz'))
    }
    Add-Table $s $rows @(220,95,135,135,130,149) 136 250 17
    [void](Add-Text $s "Recommended: K-Means`r`nWard leads DBI; DBSCAN excludes 304 customers. Scores are not accuracy." 48 415 864 78 21 $ink $true)
    Add-Note $s 'Measured source: outputs/model_comparison.csv. Higher silhouette/CH and lower DBI are preferred. Rank averages include a coverage penalty and require 80% coverage.'

    $s=New-Slide 'Dashboard demonstration'
    Add-PictureFit $s (Join-Path $doc 'screenshots/01_home.png') 42 104 535 404
    [void](Add-Text $s "1. Sign in as administrator.`r`n`r`n2. Inspect customer analysis.`r`n`r`n3. Compare fitted models.`r`n`r`n4. Predict and explain a profile.`r`n`r`n5. Export recommendations." 615 121 298 365 21 $ink)
    Add-Note $s 'Actual dashboard screenshot. Demo account admin/admin123 is for local college use. The original legacy app is a separate historical demo.'

    $s=New-Slide 'Results'
    Add-PictureFit $s (Join-Path $root 'outputs/customer_clusters.png') 42 120 565 346
    [void](Add-Text $s "4 K-Means groups`r`n`r`n0.373 silhouette`r`n`r`nVIP, growth, value-focused and at-risk engagement profiles`r`n`r`nSynthetic evidence; no revenue claim" 638 129 270 337 21 $ink)
    Add-Note $s 'Measured project outputs. Plot is an income/spending projection of six-feature clusters. Cluster IDs are arbitrary.'

    $s=New-Slide 'Future scope'
    Add-Body $s @('Validate on representative customer data and audit feature choices.','Study stability across samples, time windows and parameter settings.','Evaluate campaigns with controlled experiments.','Add managed identity, retention controls and monitored deployment.')

    $s=New-Slide 'Conclusion'
    Add-Body $s @('A complete, reproducible customer analytics workflow is implemented.','RFM and multiple algorithms extend the original K-Means baseline.','Profiles and explanations make outputs easier to inspect and discuss.','The next step is real-data and business-outcome validation.')
    Add-Note $s 'Do not claim accuracy, causal explanations or proven campaign success. Invite questions about assumptions, DBSCAN coverage and prediction extensions.'

    if($deck.Slides.Count -ne 15){throw 'Expected 15 slides'}
    $issues=@()
    foreach($slide in $deck.Slides){
        foreach($shape in $slide.Shapes){
            if($shape.HasTextFrame -eq -1 -and $shape.TextFrame.HasText -eq -1){
                if($shape.TextFrame2.TextRange.BoundHeight -gt $shape.Height+3){$issues+="Slide $($slide.SlideIndex): text overflow"}
            }
        }
    }
    if($issues.Count){throw ($issues -join '; ')}
    $deck.SaveAs($output,24)
    $deck.Export($render,'PNG',1280,720)
    $deck.SaveAs((Join-Path $doc '.qa/Presentation_preview.pdf'),32)
    $deck.Close()
    $reopened=$powerpoint.Presentations.Open($output,0,0,0)
    if($reopened.Slides.Count -ne 15){throw 'Reopen slide count failed'}
    $reopened.Close()
    Copy-Item -LiteralPath $output -Destination (Join-Path $doc 'Presentation.pptx') -Force
    Copy-Item -LiteralPath $output -Destination (Join-Path $root 'AI_Customer_Segmentation_Presentation.pptx') -Force
    Write-Output "Created and reopened 15 editable slides: $output"
} finally {
    if(-not $wasRunning){$powerpoint.Quit()}
    [void][Runtime.InteropServices.Marshal]::ReleaseComObject($powerpoint)
}
