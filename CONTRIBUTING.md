# Contributing to Legonizer4

Terima kasih atas minat Anda untuk berkontribusi pada Legonizer4! 🎉

## 📋 Cara Berkontribusi

### 1. Setup Development Environment

```bash
# Fork dan clone repository
git clone https://github.com/yohannes1097/legonizer4.git
cd legonizer4

# Setup virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# atau
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
pip install -e ".[dev]"
```

### 2. Development Workflow

1. **Buat branch baru**
   ```bash
   git checkout -b feature/nama-fitur
   ```

2. **Implementasi perubahan**
   - Ikuti coding standards yang ada
   - Tambahkan docstrings untuk fungsi/class baru
   - Tulis unit tests untuk kode baru

3. **Testing**
   ```bash
   # Jalankan semua tests
   python run_tests.py
   
   # Atau test spesifik
   python -m pytest tests/test_models.py -v
   ```

4. **Code Quality**
   ```bash
   # Format code dengan black
   black src/ tests/
   
   # Check dengan flake8
   flake8 src/ tests/
   
   # Type checking dengan mypy
   mypy src/
   ```

5. **Commit dan Push**
   ```bash
   git add .
   git commit -m "feat: tambah fitur xyz"
   git push origin feature/nama-fitur
   ```

6. **Buat Pull Request**

## 🎯 Areas yang Membutuhkan Kontribusi

### High Priority
- [ ] Optimasi performa model inference
- [ ] Implementasi data augmentation tambahan
- [ ] Integrasi dengan MLflow
- [ ] Web interface untuk visualisasi

### Medium Priority
- [ ] Support untuk format gambar tambahan
- [ ] Implementasi caching untuk API
- [ ] Dokumentasi API yang lebih detail
- [ ] Benchmark performance

### Low Priority
- [ ] Support untuk model arsitektur lain
- [ ] Integrasi dengan cloud storage
- [ ] Mobile app companion
- [ ] Batch processing utilities

## 📝 Coding Standards

### Python Style Guide
- Ikuti PEP 8
- Gunakan type hints
- Maksimal 88 karakter per baris (black default)
- Gunakan docstrings format Google

### Contoh Docstring
```python
def process_image(image_path: str, target_size: tuple) -> np.ndarray:
    """
    Process gambar untuk training atau inference.
    
    Args:
        image_path: Path ke file gambar
        target_size: Ukuran target (width, height)
        
    Returns:
        Processed image sebagai numpy array
        
    Raises:
        ValueError: Jika file gambar tidak valid
    """
    pass
```

### Commit Message Format
Gunakan conventional commits:
- `feat:` untuk fitur baru
- `fix:` untuk bug fixes
- `docs:` untuk dokumentasi
- `test:` untuk tests
- `refactor:` untuk refactoring
- `perf:` untuk performance improvements

## 🧪 Testing Guidelines

### Unit Tests
- Setiap modul harus memiliki unit tests
- Coverage minimal 80%
- Gunakan descriptive test names
- Mock external dependencies

### Integration Tests
- Test end-to-end workflows
- Test API endpoints
- Test dengan data real (jika memungkinkan)

### Test Structure
```python
class TestClassName(unittest.TestCase):
    def setUp(self):
        """Setup test environment"""
        pass
    
    def tearDown(self):
        """Cleanup after tests"""
        pass
    
    def test_specific_functionality(self):
        """Test specific functionality with descriptive name"""
        # Arrange
        # Act
        # Assert
        pass
```

## 📚 Dokumentasi

### Code Documentation
- Semua public functions/classes harus memiliki docstrings
- Gunakan type hints
- Dokumentasikan parameter dan return values
- Sertakan contoh penggunaan jika perlu

### README Updates
- Update README.md jika menambah fitur baru
- Tambahkan contoh penggunaan
- Update installation instructions jika perlu

## 🐛 Bug Reports

Saat melaporkan bug, sertakan:
- Deskripsi bug yang jelas
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, dll)
- Error messages/stack traces
- Screenshots jika relevan

Template bug report:
```markdown
**Bug Description**
Brief description of the bug

**To Reproduce**
1. Step 1
2. Step 2
3. Step 3

**Expected Behavior**
What should happen

**Actual Behavior**
What actually happens

**Environment**
- OS: 
- Python version:
- Legonizer4 version:

**Additional Context**
Any other relevant information
```

## 💡 Feature Requests

Template feature request:
```markdown
**Feature Description**
Clear description of the feature

**Use Case**
Why is this feature needed?

**Proposed Solution**
How should this be implemented?

**Alternatives Considered**
Other approaches considered

**Additional Context**
Any other relevant information
```

## 🔍 Code Review Process

### For Contributors
- Pastikan semua tests pass
- Update dokumentasi jika perlu
- Respond to review comments
- Keep PRs focused dan tidak terlalu besar

### For Reviewers
- Check functionality
- Review code quality
- Verify tests
- Check documentation
- Be constructive dan helpful

## 📞 Getting Help

- **GitHub Issues**: Untuk bug reports dan feature requests
- **GitHub Discussions**: Untuk pertanyaan umum
- **Email**: team@legonizer4.com

## 🏆 Recognition

Contributors akan diakui di:
- README.md
- Release notes
- Contributors page

Terima kasih atas kontribusi Anda! 🙏
