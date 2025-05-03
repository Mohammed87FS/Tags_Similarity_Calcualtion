"""
API routes for the research field similarity application.
"""

import logging
from flask import Blueprint, request, jsonify, send_file
from typing import Dict, List, Any

from services.data_service import DataService
from services.similarity.final_calculation import FieldSimilarityService
from config import SIMILARITY_FILE

logger = logging.getLogger(__name__)

# Create Blueprint
api_bp = Blueprint('api', __name__)

# Initialize services
data_service = DataService()
field_similarity_service = FieldSimilarityService()

# Load initial data to set up field mappings
nested_data, _ = data_service.load_data()
field_similarity_service.update_field_mappings(nested_data)


@api_bp.route('/add_field', methods=['POST'])
def add_field():
    """Add a new field and calculate similarities."""
    # Load current data
    nested_data, similarities = data_service.load_data()
    
    # Get form data
    field_data = {
        "name": request.form.get('name'),
        "group": request.form.get('group'),
        "subgroup": request.form.get('subgroup'),
        "definition": request.form.get('definition'),
        "methodologies": request.form.get('methodologies'),
        "applications": request.form.get('applications'),

    }
    
    # Validate required fields
    if not field_data["name"] or not field_data["group"] or not field_data["subgroup"]:
        return jsonify({"error": "Name, group, and subgroup are required"}), 400
    
    # Check if field name already exists
    field_names = data_service.get_all_field_names(nested_data)
    if field_data["name"] in field_names:
        return jsonify({"error": "Field name already exists"}), 400
    
    # Add field to data
    nested_data, new_field = data_service.add_field_to_data(nested_data, field_data)
    
    # Update field mappings
    field_similarity_service.update_field_mappings(nested_data)
    
    # Calculate new similarities
    updated_similarities = field_similarity_service.calculate_new_similarities(
        nested_data, similarities, new_field)
    
    # Save updated data
    if data_service.save_data(nested_data, updated_similarities):
        return jsonify({
            "success": True,
            "message": "Field added and similarities calculated",
            "download_ready": True
        })
    else:
        return jsonify({"error": "Error saving data"}), 500


@api_bp.route('/get_subgroups', methods=['GET'])
def get_subgroups():
    """Get subgroups for a specific group."""
    group = request.args.get('group')
    
    if not group:
        return jsonify({"error": "Group parameter is required"}), 400
    
    nested_data, _ = data_service.load_data()
    _, subgroups = data_service.get_all_groups_and_subgroups(nested_data)
    
    return jsonify({
        "success": True,
        "subgroups": subgroups.get(group, [])
    })


@api_bp.route('/get_similarity', methods=['GET'])
def get_similarity():
    """Get similarity between two fields."""
    field1 = request.args.get('field1')
    field2 = request.args.get('field2')
    
    if not field1 or not field2:
        return jsonify({"error": "Both field1 and field2 parameters are required"}), 400
    
    nested_data, similarities = data_service.load_data()
    
    # Get field data
    field1_data = data_service.get_field_data(nested_data, field1)
    field2_data = data_service.get_field_data(nested_data, field2)
    
    if not field1_data or not field2_data:
        return jsonify({"error": "One or both fields not found"}), 404
    
    # Find similarity
    similarity = data_service.find_similarity(similarities, field1, field2)
    
    if similarity is None:
        # If similarity is not found in the database, calculate it on the fly
        try:
            similarity = field_similarity_service.compare_fields(field1_data, field2_data)
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return jsonify({"error": "Error calculating similarity"}), 500
    
    return jsonify({
        "success": True,
        "field1": field1,
        "field2": field2,
        "similarity": similarity,
        "field1_data": field1_data,
        "field2_data": field2_data
    })


@api_bp.route('/get_all_similarities_for_field', methods=['GET'])
def get_all_similarities_for_field():
    """Get similarities between one field and all other fields."""
    field_name = request.args.get('field')
    
    if not field_name:
        return jsonify({"error": "Field parameter is required"}), 400
    
    nested_data, similarities = data_service.load_data()
    
    # Get the source field data
    source_field_data = data_service.get_field_data(nested_data, field_name)
    
    if not source_field_data:
        return jsonify({"error": f"Field '{field_name}' not found"}), 404
    
    # Extract all other field names
    all_field_names = data_service.get_all_field_names(nested_data)
    
    # Find all similarities involving this field
    field_similarities = []
    for other_field in all_field_names:
        if other_field == field_name:
            continue  # Skip self-comparison
            
        # Find similarity between these fields
        similarity_score = data_service.find_similarity(similarities, field_name, other_field)
        
        if similarity_score is not None:
            # Get the other field's data
            other_field_data = data_service.get_field_data(nested_data, other_field)
            
            # Get group/subgroup info
            group, subgroup = data_service.get_field_group_info(nested_data, other_field)
            
            # Add to results
            field_similarities.append({
                "field": other_field,
                "similarity": similarity_score,
                "group": group or "",
                "subgroup": subgroup or "",
                "field_data": other_field_data
            })
    
    # Sort by similarity (highest first)
    field_similarities.sort(key=lambda x: x["similarity"], reverse=True)
    
    return jsonify({
        "success": True,
        "field": field_name,
        "source_field_data": source_field_data,
        "similarities": field_similarities
    })


@api_bp.route('/download_similarities')
def download_similarities():
    """Download similarities file."""
    return send_file(SIMILARITY_FILE, as_attachment=True)


@api_bp.route('/test')
def test():
    """Test route to check if data is loading correctly."""
    nested_data, similarities = data_service.load_data()
    field_names = data_service.get_all_field_names(nested_data)
    groups, subgroups = data_service.get_all_groups_and_subgroups(nested_data)
    
    return jsonify({
        "success": True,
        "data_loaded": True,
        "field_count": len(field_names),
        "group_count": len(groups),
        "similarity_count": len(similarities),
        "fields": field_names if field_names else [],
        "groups": groups if groups else []
    })


@api_bp.route('/recalculate_similarities', methods=['POST'])
def recalculate_similarities():
    """Recalculate similarities for all fields."""
    # Load current data
    nested_data, _ = data_service.load_data()
    
    # Recalculate similarities
    try:
        similarities_list = field_similarity_service.calculate_all_similarities(nested_data)
        
        # Extract all unique field names for the tags array
        unique_tags = set()
        for sim in similarities_list:
            unique_tags.add(sim["field1"])
            unique_tags.add(sim["field2"])
        
        # Create result with tags and similarities
        updated_similarities = {
            "tags": sorted(list(unique_tags)),
            "similarities": similarities_list
        }
    except Exception as e:
        logger.error(f"Error recalculating similarities: {e}")
        return jsonify({"error": f"Error recalculating similarities: {str(e)}"}), 500
    
    # Save updated data
    if data_service.save_data(nested_data, updated_similarities):
        return jsonify({
            "success": True,
            "message": "Similarities recalculated and saved",
            "count": len(similarities_list),
            "tagCount": len(unique_tags),
            "download_ready": True
        })
    else:
        return jsonify({"error": "Error saving data"}), 500


@api_bp.route('/delete_field', methods=['POST'])
def delete_field():
    """Delete a field and recalculate all similarities."""
    try:
        # Get field name from request
        request_data = request.get_json()
        field_name = request_data.get('fieldName')
        
        if not field_name:
            return jsonify({"error": "Field name is required"}), 400
        
        # Load current data
        nested_data, similarities = data_service.load_data()
        
        # Check if field exists
        field_exists = False
        for category in nested_data.get("categories", []):
            for subgroup in category.get("subgroups", []):
                for i, field in enumerate(subgroup.get("fields", [])):
                    if field["name"] == field_name:
                        # Remove the field
                        subgroup["fields"].pop(i)
                        field_exists = True
                        break
                if field_exists:
                    break
            if field_exists:
                break
        
        if not field_exists:
            return jsonify({"error": f"Field '{field_name}' not found"}), 404
        
        # Filter out similarities involving this field
        updated_similarities = [
            sim for sim in similarities 
            if sim.get("field1") != field_name and sim.get("field2") != field_name
        ]
        
        # Calculate how many were removed
        removed_count = len(similarities) - len(updated_similarities)
        
        # Save updated data
        if data_service.save_data(nested_data, updated_similarities):
            return jsonify({
                "success": True,
                "message": f"Field '{field_name}' deleted successfully",
                "updatedCount": len(updated_similarities),
                "removedCount": removed_count
            })
        else:
            return jsonify({"error": "Error saving data"}), 500
            
    except Exception as e:
        logger.error(f"Error deleting field: {e}")
        return jsonify({"error": f"Error deleting field: {str(e)}"}), 500


@api_bp.route('/delete_field_all', methods=['POST'])
def delete_field_all():
    """Delete a field and recalculate all similarities from scratch."""
    try:
        # Get field name from request
        request_data = request.get_json()
        field_name = request_data.get('fieldName')
        
        if not field_name:
            return jsonify({"error": "Field name is required"}), 400
        
        # Load current data
        nested_data, _ = data_service.load_data()
        
        # Check if field exists and remove it
        field_exists = False
        for category in nested_data.get("categories", []):
            for subgroup in category.get("subgroups", []):
                field_indexes_to_remove = []
                
                # Find all matching fields (should be just one)
                for i, field in enumerate(subgroup.get("fields", [])):
                    if field["name"] == field_name:
                        field_indexes_to_remove.append(i)
                        field_exists = True
                
                # Remove fields in reverse order to not mess up indexing
                for index in sorted(field_indexes_to_remove, reverse=True):
                    subgroup["fields"].pop(index)
        
        if not field_exists:
            return jsonify({"error": f"Field '{field_name}' not found"}), 404
        
        # Count remaining fields
        field_count = 0
        for category in nested_data.get("categories", []):
            for subgroup in category.get("subgroups", []):
                field_count += len(subgroup.get("fields", []))
        
        logger.info(f"Deleted field '{field_name}'. {field_count} fields remaining.")
        
        # Rebuild field mappings
        field_similarity_service.update_field_mappings(nested_data)
        
        # Collect all remaining fields
        all_fields = []
        for category in nested_data.get("categories", []):
            for subgroup in category.get("subgroups", []):
                for field in subgroup.get("fields", []):
                    all_fields.append(field)
        
        # Calculate all pairwise similarities
        new_similarities = []
        comparison_count = 0
        
        for i in range(len(all_fields)):
            for j in range(i + 1, len(all_fields)):
                field1 = all_fields[i]
                field2 = all_fields[j]
                
                try:
                    similarity = field_similarity_service.compare_fields(field1, field2)
                    
                    new_similarities.append({
                        "field1": field1["name"],
                        "field2": field2["name"],
                        "similarity_score": float(similarity)  # Convert any numpy type to Python float
                    })
                    comparison_count += 1
                except Exception as e:
                    logger.error(f"Error calculating similarity between {field1['name']} and {field2['name']}: {e}")
        
        logger.info(f"Completed {comparison_count} similarity calculations")
        
        # Save updated data
        if data_service.save_data(nested_data, new_similarities):
            return jsonify({
                "success": True,
                "message": f"Field '{field_name}' deleted successfully and similarities recalculated",
                "fieldCount": field_count,
                "comparisonCount": comparison_count
            })
        else:
            return jsonify({"error": "Error saving updated data"}), 500
            
    except Exception as e:
        logger.error(f"Error deleting field: {e}")
        return jsonify({"error": f"Error deleting field: {str(e)}"}), 500