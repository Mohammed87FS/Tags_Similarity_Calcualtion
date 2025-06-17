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


api_bp = Blueprint('api', __name__)


data_service = DataService()
field_similarity_service = FieldSimilarityService()


nested_data, _ = data_service.load_data()
field_similarity_service.update_field_mappings(nested_data)


@api_bp.route('/add_field', methods=['POST'])
def add_field():
    """Add a new field and calculate similarities."""
    
    nested_data, similarities = data_service.load_data()
    
    
    field_data = {
        "name": request.form.get('name'),
        "group": request.form.get('group'),
        "subgroup": request.form.get('subgroup'),
        "definition": request.form.get('definition'),
        "methodologies": request.form.get('methodologies'),
        "applications": request.form.get('applications'),

    }
    
    
    if not field_data["name"] or not field_data["group"] or not field_data["subgroup"]:
        return jsonify({"error": "Name, group, and subgroup are required"}), 400
    
    
    field_names = data_service.get_all_field_names(nested_data)
    if field_data["name"] in field_names:
        return jsonify({"error": "Field name already exists"}), 400
    
    
    nested_data, new_field = data_service.add_field_to_data(nested_data, field_data)
    
    
    field_similarity_service.update_field_mappings(nested_data)
    
    
    updated_similarities = field_similarity_service.calculate_new_similarities(
        nested_data, similarities, new_field)
    
    
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
    
    
    field1_data = data_service.get_field_data(nested_data, field1)
    field2_data = data_service.get_field_data(nested_data, field2)
    
    if not field1_data or not field2_data:
        return jsonify({"error": "One or both fields not found"}), 404
    
    
    similarity = data_service.find_similarity(similarities, field1, field2)
    
    if similarity is None:
        
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
    
    
    source_field_data = data_service.get_field_data(nested_data, field_name)
    
    if not source_field_data:
        return jsonify({"error": f"Field '{field_name}' not found"}), 404
    
    
    all_field_names = data_service.get_all_field_names(nested_data)
    
    
    field_similarities = []
    for other_field in all_field_names:
        if other_field == field_name:
            continue  
            
        
        similarity_score = data_service.find_similarity(similarities, field_name, other_field)
        
        if similarity_score is not None:
            
            other_field_data = data_service.get_field_data(nested_data, other_field)
            
            
            group, subgroup = data_service.get_field_group_info(nested_data, other_field)
            
            
            field_similarities.append({
                "field": other_field,
                "similarity": similarity_score,
                "group": group or "",
                "subgroup": subgroup or "",
                "field_data": other_field_data
            })
    
    
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
    
    nested_data, _ = data_service.load_data()
    
    
    try:
        similarities_list = field_similarity_service.calculate_all_similarities(nested_data)
        
        
        unique_tags = set()
        for sim in similarities_list:
            unique_tags.add(sim["field1"])
            unique_tags.add(sim["field2"])
        
        
        updated_similarities = {
            "tags": sorted(list(unique_tags)),
            "similarities": similarities_list
        }
    except Exception as e:
        logger.error(f"Error recalculating similarities: {e}")
        return jsonify({"error": f"Error recalculating similarities: {str(e)}"}), 500
    
    
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
        
        request_data = request.get_json()
        field_name = request_data.get('fieldName')
        
        if not field_name:
            return jsonify({"error": "Field name is required"}), 400
        
        
        nested_data, similarities = data_service.load_data()
        
        
        field_exists = False
        for category in nested_data.get("categories", []):
            for subgroup in category.get("subgroups", []):
                for i, field in enumerate(subgroup.get("fields", [])):
                    if field["name"] == field_name:
                        
                        subgroup["fields"].pop(i)
                        field_exists = True
                        break
                if field_exists:
                    break
            if field_exists:
                break
        
        if not field_exists:
            return jsonify({"error": f"Field '{field_name}' not found"}), 404
        
        
        updated_similarities = [
            sim for sim in similarities 
            if sim.get("field1") != field_name and sim.get("field2") != field_name
        ]
        
        
        removed_count = len(similarities) - len(updated_similarities)
        
        
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
        
        request_data = request.get_json()
        field_name = request_data.get('fieldName')
        
        if not field_name:
            return jsonify({"error": "Field name is required"}), 400
        
        
        nested_data, _ = data_service.load_data()
        
        
        field_exists = False
        for category in nested_data.get("categories", []):
            for subgroup in category.get("subgroups", []):
                field_indexes_to_remove = []
                
                
                for i, field in enumerate(subgroup.get("fields", [])):
                    if field["name"] == field_name:
                        field_indexes_to_remove.append(i)
                        field_exists = True
                
                
                for index in sorted(field_indexes_to_remove, reverse=True):
                    subgroup["fields"].pop(index)
        
        if not field_exists:
            return jsonify({"error": f"Field '{field_name}' not found"}), 404
        
        
        field_count = 0
        for category in nested_data.get("categories", []):
            for subgroup in category.get("subgroups", []):
                field_count += len(subgroup.get("fields", []))
        
        logger.info(f"Deleted field '{field_name}'. {field_count} fields remaining.")
        
        
        field_similarity_service.update_field_mappings(nested_data)
        
        
        all_fields = []
        for category in nested_data.get("categories", []):
            for subgroup in category.get("subgroups", []):
                for field in subgroup.get("fields", []):
                    all_fields.append(field)
        
        
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
                        "similarity_score": float(similarity)  
                    })
                    comparison_count += 1
                except Exception as e:
                    logger.error(f"Error calculating similarity between {field1['name']} and {field2['name']}: {e}")
        
        logger.info(f"Completed {comparison_count} similarity calculations")
        
        
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