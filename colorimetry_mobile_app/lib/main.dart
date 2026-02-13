// pubspec.yaml dependencies required:
// dependencies:
//   flutter:
//     sdk: flutter
//   image_picker: ^1.0.4
//   http: ^1.1.0
//   path_provider: ^2.1.1
//   permission_handler: ^11.0.1
//   open_file: ^3.3.2

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:io';
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:open_file/open_file.dart';
import 'package:permission_handler/permission_handler.dart';

Future<void> requestStoragePermission() async {
  var status = await Permission.storage.status;
  if (!status.isGranted) {
    await Permission.storage.request();
  }
}

void main() {
  runApp(const ColorimetryApp());
}

class ColorimetryApp extends StatelessWidget {
  const ColorimetryApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Colorimetric Analysis',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        primarySwatch: Colors.blue,
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF2196F3),
          brightness: Brightness.light,
        ),
        cardTheme: CardThemeData(
          elevation: 0,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
          ),
        ),
      ),
      home: const HomeScreen(),
    );
  }
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({Key? key}) : super(key: key);

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> with SingleTickerProviderStateMixin {
  final ImagePicker _picker = ImagePicker();
  File? _imageFile;
  String _experimentType = 'Albumin';
  int _rows = 3;
  int _cols = 5;
  List<TextEditingController> _concentrationControllers = [];
  bool _isLoading = false;
  Map<String, dynamic>? _results;
  late AnimationController _animationController;
  late Animation<double> _fadeAnimation;

  // Update this with your backend URL
  final String baseUrl = 'http://192.168.1.7:8000';

  @override
  void initState() {
    super.initState();
    _initConcentrationControllers();
    _animationController = AnimationController(
      duration: const Duration(milliseconds: 500),
      vsync: this,
    );
    _fadeAnimation = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(parent: _animationController, curve: Curves.easeIn),
    );
  }

  void _initConcentrationControllers() {
    _concentrationControllers = List.generate(
      _cols - 1,
      (index) => TextEditingController(),
    );
  }

  void _updateConcentrationFields() {
    for (var controller in _concentrationControllers) {
      controller.dispose();
    }
    _initConcentrationControllers();
    setState(() {});
  }

  @override
  void dispose() {
    for (var controller in _concentrationControllers) {
      controller.dispose();
    }
    _animationController.dispose();
    super.dispose();
  }

  Future<void> _pickImage(ImageSource source) async {
    try {
      final XFile? image = await _picker.pickImage(
        source: source,
        imageQuality: 85,
      );
      if (image != null) {
        setState(() {
          _imageFile = File(image.path);
          _results = null;
        });
        _animationController.forward(from: 0.0);
      }
    } catch (e) {
      _showError('Failed to pick image: $e');
    }
  }

  Future<void> _analyzeImage() async {
    if (_imageFile == null) {
      _showError('Please select an image first');
      return;
    }

    // Validate concentration inputs
    List<double> concentrations = [];
    for (var controller in _concentrationControllers) {
      if (controller.text.isEmpty) {
        _showError('Please fill all concentration fields');
        return;
      }
      try {
        concentrations.add(double.parse(controller.text));
      } catch (e) {
        _showError('Invalid concentration value');
        return;
      }
    }

    setState(() {
      _isLoading = true;
    });

    try {
      var request = http.MultipartRequest(
        'POST',
        Uri.parse('$baseUrl/analyze'),
      );

      request.files.add(
        await http.MultipartFile.fromPath('image', _imageFile!.path),
      );

      request.fields['experiment_type'] = _experimentType;
      request.fields['rows'] = _rows.toString();
      request.fields['cols'] = _cols.toString();
      request.fields['concentrations'] = jsonEncode(concentrations);

      var streamedResponse = await request.send();
      var response = await http.Response.fromStream(streamedResponse);

     if (response.statusCode == 200) {
        setState(() {
          _results = jsonDecode(response.body);
          _isLoading = false;
        });
        _navigateToResults();
      } else {
        setState(() {
          _isLoading = false;
        });
        try {
          final errorMsg = jsonDecode(response.body)['error'];
          _showError('Analysis failed: $errorMsg');
        } catch (_) {
          _showError('Analysis failed: ${response.body}');
        }
      }
    } catch (e) {
      setState(() {
        _isLoading = false;
      });
      _showError('Network error: $e');
    }
  }

  void _navigateToResults() {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => ResultsScreen(
          results: _results!,
          baseUrl: baseUrl,
        ),
      ),
    );
  }

  void _showError(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: Colors.red,
        behavior: SnackBarBehavior.floating,
      ),
    );
  }

  void _showImageSourceDialog() {
    showModalBottomSheet(
      context: context,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (context) => SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              ListTile(
                leading: const Icon(Icons.camera_alt, color: Color(0xFF2196F3)),
                title: const Text('Camera'),
                onTap: () {
                  Navigator.pop(context);
                  _pickImage(ImageSource.camera);
                },
              ),
              ListTile(
                leading: const Icon(Icons.photo_library, color: Color(0xFF2196F3)),
                title: const Text('Gallery'),
                onTap: () {
                  Navigator.pop(context);
                  _pickImage(ImageSource.gallery);
                },
              ),
            ],
          ),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.grey[50],
      appBar: AppBar(
        title: const Text(
          'Colorimetric Analysis',
          style: TextStyle(fontWeight: FontWeight.w600),
        ),
        centerTitle: true,
        elevation: 0,
        backgroundColor: Colors.transparent,
        foregroundColor: Colors.black87,
      ),
      body: _isLoading
          ? const Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  CircularProgressIndicator(),
                  SizedBox(height: 16),
                  Text(
                    'Analyzing image...',
                    style: TextStyle(fontSize: 16, color: Colors.black54),
                  ),
                ],
              ),
            )
          : SingleChildScrollView(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  // Image Selection Card
                  Card(
                    child: InkWell(
                      onTap: _showImageSourceDialog,
                      borderRadius: BorderRadius.circular(16),
                      child: Container(
                        height: 200,
                        decoration: BoxDecoration(
                          borderRadius: BorderRadius.circular(16),
                          gradient: LinearGradient(
                            colors: [
                              Colors.blue.shade50,
                              Colors.blue.shade100,
                            ],
                            begin: Alignment.topLeft,
                            end: Alignment.bottomRight,
                          ),
                        ),
                        child: _imageFile == null
                            ? Column(
                                mainAxisAlignment: MainAxisAlignment.center,
                                children: [
                                  Icon(
                                    Icons.add_photo_alternate_outlined,
                                    size: 64,
                                    color: Colors.blue.shade300,
                                  ),
                                  const SizedBox(height: 8),
                                  Text(
                                    'Tap to select image',
                                    style: TextStyle(
                                      fontSize: 16,
                                      color: Colors.blue.shade700,
                                      fontWeight: FontWeight.w500,
                                    ),
                                  ),
                                ],
                              )
                            : FadeTransition(
                                opacity: _fadeAnimation,
                                child: ClipRRect(
                                  borderRadius: BorderRadius.circular(16),
                                  child: Image.file(
                                    _imageFile!,
                                    fit: BoxFit.cover,
                                  ),
                                ),
                              ),
                      ),
                    ),
                  ),
                  const SizedBox(height: 24),

                  // Experiment Type
                  Card(
                    child: Padding(
                      padding: const EdgeInsets.all(16),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Text(
                            'Experiment Type',
                            style: TextStyle(
                              fontSize: 16,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                          const SizedBox(height: 12),
                          DropdownButtonFormField<String>(
                            value: _experimentType,
                            decoration: InputDecoration(
                              border: OutlineInputBorder(
                                borderRadius: BorderRadius.circular(12),
                              ),
                              contentPadding: const EdgeInsets.symmetric(
                                horizontal: 16,
                                vertical: 12,
                              ),
                            ),
                            items: const [
                              DropdownMenuItem(
                                value: 'Albumin',
                                child: Text('Albumin'),
                              ),
                              DropdownMenuItem(
                                value: 'Total Protein',
                                child: Text('Total Protein'),
                              ),
                            ],
                            onChanged: (value) {
                              setState(() {
                                _experimentType = value!;
                              });
                            },
                          ),
                        ],
                      ),
                    ),
                  ),
                  const SizedBox(height: 16),

                  // Grid Configuration
                  Card(
                    child: Padding(
                      padding: const EdgeInsets.all(16),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Text(
                            'Grid Configuration',
                            style: TextStyle(
                              fontSize: 16,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                          const SizedBox(height: 16),
                          Row(
                            children: [
                              Expanded(
                                child: Column(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    const Text('Rows'),
                                    const SizedBox(height: 8),
                                    DropdownButtonFormField<int>(
                                      value: _rows,
                                      decoration: InputDecoration(
                                        border: OutlineInputBorder(
                                          borderRadius: BorderRadius.circular(12),
                                        ),
                                        contentPadding: const EdgeInsets.symmetric(
                                          horizontal: 16,
                                          vertical: 12,
                                        ),
                                      ),
                                      items: List.generate(
                                        10,
                                        (i) => DropdownMenuItem(
                                          value: i + 1,
                                          child: Text('${i + 1}'),
                                        ),
                                      ),
                                      onChanged: (value) {
                                        setState(() {
                                          _rows = value!;
                                        });
                                      },
                                    ),
                                  ],
                                ),
                              ),
                              const SizedBox(width: 16),
                              Expanded(
                                child: Column(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    const Text('Columns'),
                                    const SizedBox(height: 8),
                                    DropdownButtonFormField<int>(
                                      value: _cols,
                                      decoration: InputDecoration(
                                        border: OutlineInputBorder(
                                          borderRadius: BorderRadius.circular(12),
                                        ),
                                        contentPadding: const EdgeInsets.symmetric(
                                          horizontal: 16,
                                          vertical: 12,
                                        ),
                                      ),
                                      items: List.generate(
                                        10,
                                        (i) => DropdownMenuItem(
                                          value: i + 2,
                                          child: Text('${i + 2}'),
                                        ),
                                      ),
                                      onChanged: (value) {
                                        setState(() {
                                          _cols = value!;
                                          _updateConcentrationFields();
                                        });
                                      },
                                    ),
                                  ],
                                ),
                              ),
                            ],
                          ),
                        ],
                      ),
                    ),
                  ),
                  const SizedBox(height: 16),

                  // Concentrations
                  Card(
                    child: Padding(
                      padding: const EdgeInsets.all(16),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Text(
                            'Concentrations',
                            style: TextStyle(
                              fontSize: 16,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                          const SizedBox(height: 8),
                          Text(
                            'Enter ${_cols - 1} concentration values in g/dL (blank excluded)',
                            style: TextStyle(
                              fontSize: 12,
                              color: Colors.grey[600],
                            ),
                          ),
                          const SizedBox(height: 16),
                          ...List.generate(
                            _concentrationControllers.length,
                            (index) => Padding(
                              padding: const EdgeInsets.only(bottom: 12),
                              child: TextFormField(
                                controller: _concentrationControllers[index],
                                keyboardType: const TextInputType.numberWithOptions(
                                  decimal: true,
                                ),
                                decoration: InputDecoration(
                                  labelText: 'Sample ${index + 1}',
                                  border: OutlineInputBorder(
                                    borderRadius: BorderRadius.circular(12),
                                  ),
                                  contentPadding: const EdgeInsets.symmetric(
                                    horizontal: 16,
                                    vertical: 12,
                                  ),
                                ),
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                  const SizedBox(height: 24),

                  // Analyze Button
                  ElevatedButton(
                    onPressed: _imageFile != null ? _analyzeImage : null,
                    style: ElevatedButton.styleFrom(
                      backgroundColor: const Color(0xFF2196F3),
                      foregroundColor: Colors.white,
                      padding: const EdgeInsets.symmetric(vertical: 16),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12),
                      ),
                      elevation: 0,
                    ),
                    child: const Text(
                      'Analyze Image',
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ),
                  const SizedBox(height: 16),
                ],
              ),
            ),
    );
  }
}

class ResultsScreen extends StatelessWidget {
  final Map<String, dynamic> results;
  final String baseUrl;
  
  

  const ResultsScreen({
    Key? key,
    required this.results,
    required this.baseUrl,
  }) : super(key: key);

  Future<void> _downloadFile(BuildContext context, String filePath) async {
  try {
    bool granted = false;
    if (Platform.isAndroid) {
      // Android 13+ (API 33) uses photos/media permissions
      var storageStatus = await Permission.storage.request();
      var photosStatus = await Permission.photos.request();
      var mediaStatus = await Permission.mediaLibrary.request();
      granted = storageStatus.isGranted || photosStatus.isGranted || mediaStatus.isGranted;
    } else if (Platform.isIOS) {
      var photosStatus = await Permission.photos.request();
      granted = photosStatus.isGranted;
    } else {
      granted = true; // Other platforms
    }

    if (!granted) {
      throw 'Storage or media permission denied';
    }

    final url = '$baseUrl/download/$filePath';
    final response = await http.get(Uri.parse(url));

    if (response.statusCode == 200) {
      Directory? dir;
      if (Platform.isAndroid) {
        dir = await getExternalStorageDirectory();
      } else if (Platform.isIOS) {
        dir = await getApplicationDocumentsDirectory();
      } else {
        dir = await getTemporaryDirectory();
      }
      final fileName = filePath.split('/').last;
      final file = File('${dir!.path}/$fileName');
      await file.writeAsBytes(response.bodyBytes);

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Downloaded: $fileName'),
          backgroundColor: Colors.green,
          behavior: SnackBarBehavior.floating,
          action: SnackBarAction(
            label: 'Open',
            textColor: Colors.white,
            onPressed: () {
              OpenFile.open(file.path);
            },
          ),
        ),
      );
    } else {
      throw 'Download failed';
    }
  } catch (e) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text('Error: $e'),
        backgroundColor: Colors.red,
        behavior: SnackBarBehavior.floating,
      ),
    );
  }
}

  @override
  Widget build(BuildContext context) {
    final plots = results['plots'] as List;
    final datasets = results['datasets'] as Map<String, dynamic>;
    print(datasets);


    return Scaffold(
      backgroundColor: Colors.grey[50],
      appBar: AppBar(
        title: const Text(
          'Analysis Results',
          style: TextStyle(fontWeight: FontWeight.w600),
        ),
        centerTitle: true,
        elevation: 0,
        backgroundColor: Colors.transparent,
        foregroundColor: Colors.black87,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Plots Section
            const Text(
              'Generated Plots',
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 16),
            ...List.generate(plots.length, (index) {
              final plotPath = plots[index];
              final plotUrl = '$baseUrl/download/$plotPath';
              final titles = [
                'Individual Extracted Circles',
                'Circles on Original Image',
                'Value vs Concentration',
                'Linear Regression',
              ];

              return Card(
                margin: const EdgeInsets.only(bottom: 16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    Padding(
                      padding: const EdgeInsets.all(16),
                      child: Text(
                        titles[index],
                        style: const TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ),
                    Image.network(
                      plotUrl,
                      fit: BoxFit.contain,
                      loadingBuilder: (context, child, loadingProgress) {
                        if (loadingProgress == null) return child;
                        return const Center(
                          child: Padding(
                            padding: EdgeInsets.all(32),
                            child: CircularProgressIndicator(),
                          ),
                        );
                      },
                      errorBuilder: (context, error, stackTrace) {
                        return const Padding(
                          padding: EdgeInsets.all(32),
                          child: Center(
                            child: Text('Failed to load image'),
                          ),
                        );
                      },
                    ),
                    Padding(
                      padding: const EdgeInsets.all(16),
                      child: OutlinedButton.icon(
                        onPressed: () => _downloadFile(context, plotPath),
                        icon: const Icon(Icons.download),
                        label: const Text('Download'),
                        style: OutlinedButton.styleFrom(
                          padding: const EdgeInsets.symmetric(vertical: 12),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(12),
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
              );
            }),
            const SizedBox(height: 24),

            // Datasets Section
            const Text(
              'Datasets',
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 16),
            Card(
              child: ListTile(
                leading: const Icon(Icons.table_chart, color: Color(0xFF2196F3)),
                title: const Text('All Color Data'),
                subtitle: const Text('Complete color parameters'),
                trailing: IconButton(
                  icon: const Icon(Icons.download),
                  onPressed: () => _downloadFile(
                    context,
                    datasets['all_color_data'],
                  ),
                ),
              ),
            ),
            const SizedBox(height: 8),
            Card(
              child: ListTile(
                leading: const Icon(Icons.analytics, color: Color(0xFF2196F3)),
                title: const Text('Linear Regression Ranking'),
                subtitle: const Text('Regression analysis results'),
                trailing: IconButton(
                  icon: const Icon(Icons.download),
                  onPressed: () => _downloadFile(
                    context,
                    datasets['linear_regression_ranking'],
                  ),
                ),
              ),
            ),

            const SizedBox(height: 24),
          if (datasets['lod_table'] != null)
            Card(
              child: ListTile(
                leading: const Icon(Icons.science, color: Color(0xFF2196F3)),
                title: const Text('LOD Table'),
                subtitle: const Text('Limit of Detection analysis'),
                trailing: IconButton(
                  icon: const Icon(Icons.download),
                  onPressed: () => _downloadFile(
                    context,
                    datasets['lod_table'],
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}